package com.example.shopping.sensor

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class PdrPoint(val x: Float, val y: Float, val stepIndex: Int)

data class PdrSnapshot(
    val segmentSteps: Int,
    val segmentDistanceM: Float,
    val avgHeadingDeg: Float,
    val currentX: Float,
    val currentY: Float,
    val totalSteps: Int,
    val totalDistanceM: Float,
    val rawSensorJson: JSONObject,
    val gameRotVecYawDeg: Float,
    val rotVecYawDeg: Float,
    val azimuthDeg: Float,
    val complementaryYawDeg: Float,
    val gyroOnlyYawDeg: Float,
    val calibratedYawDeg: Float,
    val gameRotVecQuat: FloatArray,
    val rotVecQuat: FloatArray,
    val gpsLat: Double?,
    val gpsLng: Double?,
    val gpsAccuracy: Float?,
)

class PdrTracker(private val sensorManager: SensorManager) : SensorEventListener {

    // ── Magnetometer calibration status ──

    var magAccuracy: Int = SensorManager.SENSOR_STATUS_UNRELIABLE; private set
    var onMagAccuracyChanged: ((Int) -> Unit)? = null

    // ── Magnetic field quality assessment ──

    enum class MagQuality { UNKNOWN, GOOD, SUSPECT, BAD }

    var magFieldStrength: Float = 0f; private set
    var magQuality: MagQuality = MagQuality.UNKNOWN; private set
    var onMagQualityChanged: ((MagQuality, Float) -> Unit)? = null

    // ── Gyroscope bias calibration ──

    enum class GyroCalState { UNCALIBRATED, CALIBRATING, CALIBRATED, FAILED }

    var gyroCalState: GyroCalState = GyroCalState.UNCALIBRATED; private set
    var gyroBias: FloatArray = floatArrayOf(0f, 0f, 0f); private set
    var gyroCalProgress: Float = 0f; private set
    var gyroCalNoise: Float = 0f; private set
    var onGyroCalStateChanged: ((GyroCalState) -> Unit)? = null

    private var gyroCalSamples = mutableListOf<FloatArray>()
    private var gyroCalStartMs = 0L
    private val gyroCalDurationMs = 3000L

    // ── Public read-only state ──

    var x: Float = 0f; private set
    var y: Float = 0f; private set
    var headingDeg: Float = 0f; private set
    var totalSteps: Int = 0; private set
    var totalDistance: Float = 0f; private set
    var lastStride: Float = 0f; private set

    private val _pathPoints = mutableListOf<PdrPoint>()
    val pathPoints: List<PdrPoint> get() = _pathPoints

    // ── Live sensor values (for SensorLab display) ──

    var accelValues: FloatArray = floatArrayOf(0f, 0f, 0f); private set
    var gyroValues: FloatArray = floatArrayOf(0f, 0f, 0f); private set
    var magValues: FloatArray = floatArrayOf(0f, 0f, 0f); private set
    var gravityValues: FloatArray = floatArrayOf(0f, 0f, 0f); private set
    var linAccValues: FloatArray = floatArrayOf(0f, 0f, 0f); private set
    var pressureValue: Float = 0f; private set
    var lightValue: Float = 0f; private set
    var stepDetectorCount: Int = 0; private set
    val currentGameRotVec: FloatArray get() = gameRotVec.copyOf()
    val currentRotVec: FloatArray get() = rotVec.copyOf()
    val gpsLat: Double? get() = lastGpsLat
    val gpsLng: Double? get() = lastGpsLng
    val gpsAccuracy: Float? get() = lastGpsAccuracy
    var onSampleReady: (() -> Unit)? = null

    // ── Internal state ──

    private var running = false
    private var weinbergK = 0.4667f
    private var startTimeMs = 0L

    private var stepBaseline = -1
    private var accelMagMax = 0f
    private var accelMagMin = Float.MAX_VALUE

    private var gameRotVec = floatArrayOf(0f, 0f, 0f, 1f)
    private var rotVec = floatArrayOf(0f, 0f, 0f, 1f)

    // ── Complementary filter (gyro short-term + mag long-term) ──
    private var compHeadingRad = 0.0
    private var compInitialized = false
    private var lastGyroTimeMs = 0L

    // ── Gyro-only heading (accel for tilt compensation via GRV initial, gyro Z integration) ──
    private var gyroOnlyYawRad = 0.0
    private var gyroOnlyInitialized = false

    // ── Startup calibration (first Rot Vec yaw → north offset for Game Rot Vec) ──
    private var initialRotVecYawRad: Double? = null
    private var initialGameRotVecYawRad: Double? = null
    val isNorthCalibrated: Boolean get() = initialRotVecYawRad != null
    var onNorthCalibrated: (() -> Unit)? = null

    // Segment tracking (between snapshots)
    private var segmentSteps = 0
    private var segmentDistance = 0f
    private var segmentHeadingSum = 0.0
    private var segmentHeadingSamples = 0

    // Heading from accelerometer + magnetometer (for display)
    private val gravity = floatArrayOf(0f, 0f, 0f)
    private val geomagnetic = floatArrayOf(0f, 0f, 0f)
    private val rotationMatrix = FloatArray(9)
    private val orientation = FloatArray(3)

    // ── Raw sensor buffers (flushed on each snapshot) ──
    // Format: [elapsed_ms, x, y, z] per sample
    private val accelBuf = mutableListOf<FloatArray>()
    private val gyroBuf = mutableListOf<FloatArray>()
    private val magBuf = mutableListOf<FloatArray>()
    // Rotation vectors: [elapsed_ms, x, y, z, w]
    private val rotVecBuf = mutableListOf<FloatArray>()
    private val gameRotVecBuf = mutableListOf<FloatArray>()
    // Step detector: [elapsed_ms]
    private val stepEventBuf = mutableListOf<Long>()
    // Step counter triggers (actual onStepDetected calls from hw): [elapsed_ms]
    private val stepCounterBuf = mutableListOf<Long>()
    // GPS: [elapsed_ms, lat, lng, accuracy, altitude]
    private val gpsBuf = mutableListOf<DoubleArray>()
    private var lastGpsLat: Double? = null
    private var lastGpsLng: Double? = null
    private var lastGpsAccuracy: Float? = null

    // Throttle: don't buffer faster than ~50Hz per sensor
    private var lastAccelBufTime = 0L
    private var lastGyroBufTime = 0L
    private var lastMagBufTime = 0L

    // ── Software step detection (fallback when hardware counter is silent) ──
    private var filteredMag = 9.81f
    private var avgMag = 9.81f
    private var wasAbove = false
    private var lastSwStepTime = 0L
    private var lastHwStepTime = 0L
    private var swStepCount = 0
    private val swStepEventBuf = mutableListOf<Long>()
    companion object {
        private const val LP_ALPHA = 0.15f
        private const val AVG_ALPHA = 0.005f
        private const val PEAK_THRESHOLD = 1.2f
        private const val MIN_STEP_MS = 280L
        private const val MAX_STEP_MS = 2000L
        private const val HW_SILENT_MS = 2500L
        private const val COMP_ALPHA = 0.98f
        private const val GYRO_CAL_NOISE_THRESHOLD = 0.05f
        private const val MAG_EXPECTED_MIN_UT = 25f
        private const val MAG_EXPECTED_MAX_UT = 65f
        private const val MAG_SUSPECT_RATIO = 1.5f
        private const val MAG_BAD_RATIO = 2.0f
    }

    // ── Public API ──

    fun start(k: Float = 0.4667f) {
        if (running) return
        weinbergK = k
        running = true
        startTimeMs = SystemClock.elapsedRealtime()

        val rate = SensorManager.SENSOR_DELAY_GAME
        val sensorTypes = listOf(
            Sensor.TYPE_ACCELEROMETER to "Accelerometer",
            Sensor.TYPE_GYROSCOPE to "Gyroscope",
            Sensor.TYPE_MAGNETIC_FIELD to "Magnetometer",
            Sensor.TYPE_STEP_COUNTER to "StepCounter",
            Sensor.TYPE_STEP_DETECTOR to "StepDetector",
            Sensor.TYPE_ROTATION_VECTOR to "RotationVector",
            Sensor.TYPE_GAME_ROTATION_VECTOR to "GameRotationVector",
            Sensor.TYPE_GRAVITY to "Gravity",
            Sensor.TYPE_LINEAR_ACCELERATION to "LinearAcceleration",
            Sensor.TYPE_PRESSURE to "Pressure",
            Sensor.TYPE_LIGHT to "Light",
        )
        sensorTypes.forEach { (type, name) ->
            val sensor = sensorManager.getDefaultSensor(type)
            if (sensor != null) {
                sensorManager.registerListener(this, sensor, rate)
                Log.i("PdrTracker", "Registered: $name")
            } else {
                Log.w("PdrTracker", "NOT AVAILABLE: $name")
            }
        }
    }

    fun stop() {
        if (!running) return
        running = false
        sensorManager.unregisterListener(this)
    }

    fun reset() {
        x = 0f; y = 0f
        headingDeg = 0f
        totalSteps = 0; totalDistance = 0f; lastStride = 0f
        stepBaseline = -1
        accelMagMax = 0f; accelMagMin = Float.MAX_VALUE
        segmentSteps = 0; segmentDistance = 0f
        segmentHeadingSum = 0.0; segmentHeadingSamples = 0
        filteredMag = 9.81f; avgMag = 9.81f; wasAbove = false
        lastSwStepTime = 0L; lastHwStepTime = 0L; swStepCount = 0
        compHeadingRad = 0.0; compInitialized = false; lastGyroTimeMs = 0L
        gyroOnlyYawRad = 0.0; gyroOnlyInitialized = false
        gyroCalState = GyroCalState.UNCALIBRATED; gyroCalProgress = 0f; gyroCalNoise = 0f
        gyroBias = floatArrayOf(0f, 0f, 0f); gyroCalSamples.clear()
        initialRotVecYawRad = null; initialGameRotVecYawRad = null
        magFieldStrength = 0f; magQuality = MagQuality.UNKNOWN
        lastGpsLat = null; lastGpsLng = null; lastGpsAccuracy = null
        accelValues = floatArrayOf(0f, 0f, 0f)
        gyroValues = floatArrayOf(0f, 0f, 0f)
        magValues = floatArrayOf(0f, 0f, 0f)
        gravityValues = floatArrayOf(0f, 0f, 0f)
        linAccValues = floatArrayOf(0f, 0f, 0f)
        pressureValue = 0f; lightValue = 0f
        stepDetectorCount = 0
        _pathPoints.clear()
        clearBuffers()
        startTimeMs = SystemClock.elapsedRealtime()
    }

    fun setWeinbergK(k: Float) {
        if (k > 0) weinbergK = k
    }

    fun startGyroCalibration() {
        gyroCalState = GyroCalState.CALIBRATING
        gyroCalSamples.clear()
        gyroCalProgress = 0f
        gyroCalStartMs = SystemClock.elapsedRealtime()
        onGyroCalStateChanged?.invoke(gyroCalState)
    }

    fun updateLocation(lat: Double, lng: Double, accuracy: Float, altitude: Double) {
        if (!running) return
        val elapsed = SystemClock.elapsedRealtime() - startTimeMs
        gpsBuf.add(doubleArrayOf(elapsed.toDouble(), lat, lng, accuracy.toDouble(), altitude))
        lastGpsLat = lat
        lastGpsLng = lng
        lastGpsAccuracy = accuracy

    }

    // Android quaternion yaw follows right-hand rule (CCW positive),
    // negate to get navigation convention (CW positive, north=0 east=90)
    private fun quatToYawRad(q: FloatArray): Double {
        return -atan2(
            2.0 * (q[3] * q[2] + q[0] * q[1]),
            1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        )
    }

    private fun quatToYawDeg(q: FloatArray): Float {
        return Math.toDegrees(quatToYawRad(q)).toFloat()
    }

    private fun calibratedYawRad(): Double {
        val initRv = initialRotVecYawRad ?: return quatToYawRad(gameRotVec)
        val initGrv = initialGameRotVecYawRad ?: return quatToYawRad(gameRotVec)
        return quatToYawRad(gameRotVec) - initGrv + initRv
    }

    private fun calibratedYawDeg(): Float =
        Math.toDegrees(calibratedYawRad()).toFloat()

    fun snapshot(): PdrSnapshot {
        val avgHeading = if (segmentHeadingSamples > 0)
            (segmentHeadingSum / segmentHeadingSamples).toFloat()
        else headingDeg

        val rawJson = buildRawSensorJson()

        val grvYaw = quatToYawDeg(gameRotVec)
        val rvYaw = quatToYawDeg(rotVec)
        val compYaw = Math.toDegrees(compHeadingRad).toFloat()
        val gyroYaw = Math.toDegrees(gyroOnlyYawRad).toFloat()
        val calYaw = calibratedYawDeg()

        val snap = PdrSnapshot(
            segmentSteps = segmentSteps,
            segmentDistanceM = segmentDistance,
            avgHeadingDeg = avgHeading,
            currentX = x,
            currentY = y,
            totalSteps = totalSteps,
            totalDistanceM = totalDistance,
            rawSensorJson = rawJson,
            gameRotVecYawDeg = grvYaw,
            rotVecYawDeg = rvYaw,
            azimuthDeg = headingDeg,
            complementaryYawDeg = compYaw,
            gyroOnlyYawDeg = gyroYaw,
            calibratedYawDeg = calYaw,
            gameRotVecQuat = gameRotVec.copyOf(),
            rotVecQuat = rotVec.copyOf(),
            gpsLat = lastGpsLat,
            gpsLng = lastGpsLng,
            gpsAccuracy = lastGpsAccuracy,
        )

        segmentSteps = 0
        segmentDistance = 0f
        segmentHeadingSum = 0.0
        segmentHeadingSamples = 0
        clearBuffers()

        return snap
    }

    // ── Public heading getters ──

    fun getGameRotVecYawDeg(): Float = quatToYawDeg(gameRotVec)
    fun getRotVecYawDeg(): Float = quatToYawDeg(rotVec)
    fun getAzimuthDeg(): Float = headingDeg
    fun getComplementaryYawDeg(): Float = Math.toDegrees(compHeadingRad).toFloat()
    fun getGyroOnlyYawDeg(): Float = Math.toDegrees(gyroOnlyYawRad).toFloat()
    fun getCalibratedYawDeg(): Float = calibratedYawDeg()

    // ── SensorEventListener ──

    override fun onSensorChanged(event: SensorEvent) {
        if (!running) return
        val elapsed = SystemClock.elapsedRealtime() - startTimeMs

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accelValues = event.values.copyOf()
                gravity[0] = event.values[0]
                gravity[1] = event.values[1]
                gravity[2] = event.values[2]

                val mag = sqrt(
                    event.values[0] * event.values[0] +
                    event.values[1] * event.values[1] +
                    event.values[2] * event.values[2]
                )
                if (mag > accelMagMax) accelMagMax = mag
                if (mag < accelMagMin) accelMagMin = mag

                if (SensorManager.getRotationMatrix(rotationMatrix, null, gravity, geomagnetic)) {
                    SensorManager.getOrientation(rotationMatrix, orientation)
                    headingDeg = Math.toDegrees(orientation[0].toDouble()).toFloat()
                    if (headingDeg < 0) headingDeg += 360f
                }

                // Software step detection: low-pass filter + peak detection
                filteredMag = LP_ALPHA * mag + (1 - LP_ALPHA) * filteredMag
                avgMag = AVG_ALPHA * mag + (1 - AVG_ALPHA) * avgMag
                val isAbove = filteredMag > avgMag + PEAK_THRESHOLD
                if (wasAbove && !isAbove) {
                    val dt = elapsed - lastSwStepTime
                    if (dt in MIN_STEP_MS..MAX_STEP_MS) {
                        val hwSilent = elapsed - lastHwStepTime > HW_SILENT_MS
                        if (hwSilent) {
                            swStepCount++
                            swStepEventBuf.add(elapsed)
                            stepCounterBuf.add(elapsed)
                            onStepDetected()
                        }
                    }
                    lastSwStepTime = elapsed
                }
                wasAbove = isAbove

                if (elapsed - lastAccelBufTime >= 20) {
                    accelBuf.add(floatArrayOf(elapsed.toFloat(), event.values[0], event.values[1], event.values[2]))
                    lastAccelBufTime = elapsed
                    onSampleReady?.invoke()
                }
            }

            Sensor.TYPE_GYROSCOPE -> {
                // Gyro bias calibration: collect samples while stationary
                if (gyroCalState == GyroCalState.CALIBRATING) {
                    gyroCalSamples.add(event.values.copyOf())
                    val elapsedCal = SystemClock.elapsedRealtime() - gyroCalStartMs
                    gyroCalProgress = (elapsedCal.toFloat() / gyroCalDurationMs).coerceIn(0f, 1f)
                    if (elapsedCal >= gyroCalDurationMs) {
                        val n = gyroCalSamples.size
                        if (n > 0) {
                            val bx = gyroCalSamples.map { it[0] }.average().toFloat()
                            val by = gyroCalSamples.map { it[1] }.average().toFloat()
                            val bz = gyroCalSamples.map { it[2] }.average().toFloat()
                            val varX = gyroCalSamples.map { (it[0] - bx).let { d -> d * d } }.average()
                            val varY = gyroCalSamples.map { (it[1] - by).let { d -> d * d } }.average()
                            val varZ = gyroCalSamples.map { (it[2] - bz).let { d -> d * d } }.average()
                            val totalNoise = sqrt((varX + varY + varZ).toFloat())
                            gyroCalNoise = totalNoise
                            if (totalNoise > GYRO_CAL_NOISE_THRESHOLD) {
                                Log.w("PdrTracker", "Gyro cal FAILED: noise=%.6f > threshold=%.4f n=$n".format(totalNoise, GYRO_CAL_NOISE_THRESHOLD))
                                gyroCalState = GyroCalState.FAILED
                            } else {
                                gyroBias = floatArrayOf(bx, by, bz)
                                Log.i("PdrTracker", "Gyro calibrated: bias=(%.6f, %.6f, %.6f) noise=%.6f n=$n".format(bx, by, bz, totalNoise))
                                gyroCalState = GyroCalState.CALIBRATED
                            }
                        } else {
                            gyroCalState = GyroCalState.FAILED
                        }
                        gyroCalProgress = 1f
                        gyroCalSamples.clear()
                        onGyroCalStateChanged?.invoke(gyroCalState)
                    }
                }

                val corrX = event.values[0] - gyroBias[0]
                val corrY = event.values[1] - gyroBias[1]
                val corrZ = event.values[2] - gyroBias[2]
                gyroValues = floatArrayOf(corrX, corrY, corrZ)

                if (lastGyroTimeMs > 0) {
                    val dtSec = (elapsed - lastGyroTimeMs) / 1000.0
                    if (dtSec in 0.001..0.2) {
                        val gyroYawRate = -corrZ.toDouble()

                        if (gyroOnlyInitialized) {
                            gyroOnlyYawRad += gyroYawRate * dtSec
                        } else {
                            gyroOnlyYawRad = quatToYawRad(gameRotVec)
                            gyroOnlyInitialized = true
                        }

                        if (compInitialized) {
                            val magRad = Math.toRadians(headingDeg.toDouble())
                            val gyroPred = compHeadingRad + gyroYawRate * dtSec
                            val diff = ((magRad - gyroPred + Math.PI) % (2 * Math.PI)) - Math.PI
                            compHeadingRad = gyroPred + (1 - COMP_ALPHA) * diff
                        } else {
                            compHeadingRad = Math.toRadians(headingDeg.toDouble())
                            compInitialized = true
                        }
                    }
                }
                lastGyroTimeMs = elapsed

                if (elapsed - lastGyroBufTime >= 20) {
                    gyroBuf.add(floatArrayOf(elapsed.toFloat(), corrX, corrY, corrZ))
                    lastGyroBufTime = elapsed
                }
            }

            Sensor.TYPE_MAGNETIC_FIELD -> {
                magValues = event.values.copyOf()
                geomagnetic[0] = event.values[0]
                geomagnetic[1] = event.values[1]
                geomagnetic[2] = event.values[2]

                val mx = event.values[0]; val my = event.values[1]; val mz = event.values[2]
                magFieldStrength = sqrt(mx * mx + my * my + mz * mz)
                val midExpected = (MAG_EXPECTED_MIN_UT + MAG_EXPECTED_MAX_UT) / 2f
                val ratio = if (magFieldStrength > midExpected) magFieldStrength / MAG_EXPECTED_MAX_UT
                            else MAG_EXPECTED_MIN_UT / magFieldStrength.coerceAtLeast(1f)
                val newQuality = when {
                    ratio >= MAG_BAD_RATIO -> MagQuality.BAD
                    ratio >= MAG_SUSPECT_RATIO -> MagQuality.SUSPECT
                    magFieldStrength in MAG_EXPECTED_MIN_UT..MAG_EXPECTED_MAX_UT -> MagQuality.GOOD
                    else -> MagQuality.SUSPECT
                }
                if (newQuality != magQuality) {
                    magQuality = newQuality
                    Log.i("PdrTracker", "Mag quality: $newQuality (${magFieldStrength.toInt()} μT, expected ${MAG_EXPECTED_MIN_UT.toInt()}-${MAG_EXPECTED_MAX_UT.toInt()})")
                    onMagQualityChanged?.invoke(newQuality, magFieldStrength)
                }

                if (elapsed - lastMagBufTime >= 20) {
                    magBuf.add(floatArrayOf(elapsed.toFloat(), event.values[0], event.values[1], event.values[2]))
                    lastMagBufTime = elapsed
                }
            }

            Sensor.TYPE_ROTATION_VECTOR -> {
                val v = event.values
                val w = if (v.size > 3) v[3]
                else sqrt((1f - v[0] * v[0] - v[1] * v[1] - v[2] * v[2]).coerceAtLeast(0f))
                rotVec = floatArrayOf(v[0], v[1], v[2], w)
                rotVecBuf.add(floatArrayOf(elapsed.toFloat(), v[0], v[1], v[2], w))
                if (initialRotVecYawRad == null) {
                    if (magAccuracy >= SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM) {
                        initialRotVecYawRad = quatToYawRad(rotVec)
                        Log.i("PdrTracker", "Initial RotVec yaw: ${"%.1f".format(Math.toDegrees(initialRotVecYawRad!!))}° (magAccuracy=$magAccuracy)")
                        onNorthCalibrated?.invoke()
                    }
                }
            }

            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                val v = event.values
                val w = if (v.size > 3) v[3]
                else sqrt((1f - v[0] * v[0] - v[1] * v[1] - v[2] * v[2]).coerceAtLeast(0f))
                gameRotVec = floatArrayOf(v[0], v[1], v[2], w)
                gameRotVecBuf.add(floatArrayOf(elapsed.toFloat(), v[0], v[1], v[2], w))
                if (initialGameRotVecYawRad == null) {
                    initialGameRotVecYawRad = quatToYawRad(gameRotVec)
                    Log.i("PdrTracker", "Initial GameRotVec yaw: ${"%.1f".format(Math.toDegrees(initialGameRotVecYawRad!!))}°")
                }
            }

            Sensor.TYPE_STEP_DETECTOR -> {
                stepDetectorCount++
                stepEventBuf.add(elapsed)
                stepCounterBuf.add(elapsed)
                onStepDetected()
                lastHwStepTime = elapsed
            }

            Sensor.TYPE_STEP_COUNTER -> {
                val raw = event.values[0].toInt()
                if (stepBaseline < 0) {
                    stepBaseline = raw
                    Log.i("PdrTracker", "StepCounter baseline set: $raw")
                }
                val hwStepCount = raw - stepBaseline
                if (hwStepCount > totalSteps) {
                    val missed = hwStepCount - totalSteps
                    Log.d("PdrTracker", "Counter catch-up: $missed steps (counter=$hwStepCount, tracked=$totalSteps)")
                    repeat(missed) {
                        stepCounterBuf.add(elapsed)
                        onStepDetected()
                    }
                }
                lastHwStepTime = elapsed
            }

            Sensor.TYPE_GRAVITY -> {
                gravityValues = event.values.copyOf()
            }

            Sensor.TYPE_LINEAR_ACCELERATION -> {
                linAccValues = event.values.copyOf()
            }

            Sensor.TYPE_PRESSURE -> {
                pressureValue = event.values[0]
            }

            Sensor.TYPE_LIGHT -> {
                lightValue = event.values[0]
            }
        }
    }

    private fun onStepDetected() {
        val accelDiff = (accelMagMax - accelMagMin).coerceAtLeast(0.1f)
        val stride = weinbergK * sqrt(sqrt(accelDiff))
        lastStride = stride

        val yaw = calibratedYawRad()

        totalSteps++
        x += (stride * sin(yaw)).toFloat()
        y += (stride * kotlin.math.cos(yaw)).toFloat()
        totalDistance += stride

        segmentSteps++
        segmentDistance += stride
        segmentHeadingSum += Math.toDegrees(yaw)
        segmentHeadingSamples++

        _pathPoints.add(PdrPoint(x, y, totalSteps))

        accelMagMax = 0f
        accelMagMin = Float.MAX_VALUE
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        if (sensor?.type == Sensor.TYPE_MAGNETIC_FIELD && accuracy != magAccuracy) {
            magAccuracy = accuracy
            Log.i("PdrTracker", "Magnetometer accuracy changed: $accuracy")
            onMagAccuracyChanged?.invoke(accuracy)
        }
    }

    // ── Raw sensor JSON builder ──

    private fun buildRawSensorJson(): JSONObject {
        val json = JSONObject()

        json.put("accel", toJsonArray(accelBuf, 4))
        json.put("gyro", toJsonArray(gyroBuf, 4))
        json.put("mag", toJsonArray(magBuf, 4))
        json.put("rot_vec", toJsonArray(rotVecBuf, 5))
        json.put("game_rot_vec", toJsonArray(gameRotVecBuf, 5))

        val steps = JSONArray()
        for (t in stepEventBuf) steps.put(t)
        json.put("step_events", steps)

        val swSteps = JSONArray()
        for (t in swStepEventBuf) swSteps.put(t)
        json.put("sw_step_events", swSteps)

        val scSteps = JSONArray()
        for (t in stepCounterBuf) scSteps.put(t)
        json.put("step_counter_events", scSteps)

        val gpsArr = JSONArray()
        for (row in gpsBuf) {
            val r = JSONArray()
            r.put(row[0].toLong())
            r.put(row[1])  // lat
            r.put(row[2])  // lng
            r.put("%.1f".format(row[3]).toDouble())  // accuracy
            r.put("%.1f".format(row[4]).toDouble())  // altitude
            gpsArr.put(r)
        }
        json.put("gps", gpsArr)

        json.put("sample_counts", JSONObject().apply {
            put("accel", accelBuf.size)
            put("gyro", gyroBuf.size)
            put("mag", magBuf.size)
            put("rot_vec", rotVecBuf.size)
            put("game_rot_vec", gameRotVecBuf.size)
            put("step_events", stepEventBuf.size)
            put("sw_step_events", swStepEventBuf.size)
            put("step_counter_events", stepCounterBuf.size)
            put("gps", gpsBuf.size)
        })

        return json
    }

    private fun toJsonArray(buf: List<FloatArray>, cols: Int): JSONArray {
        val arr = JSONArray()
        for (row in buf) {
            val r = JSONArray()
            for (i in 0 until cols.coerceAtMost(row.size)) {
                r.put(if (i == 0) row[i].toLong() else "%.4f".format(row[i]).toDouble())
            }
            arr.put(r)
        }
        return arr
    }

    private fun clearBuffers() {
        accelBuf.clear()
        gyroBuf.clear()
        magBuf.clear()
        rotVecBuf.clear()
        gameRotVecBuf.clear()
        stepEventBuf.clear()
        swStepEventBuf.clear()
        stepCounterBuf.clear()
        gpsBuf.clear()
        lastAccelBufTime = 0L
        lastGyroBufTime = 0L
        lastMagBufTime = 0L
    }
}
