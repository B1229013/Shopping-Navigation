package com.example.shopping.sensor

import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.FileProvider
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

data class LocalSession(
    val id: String,
    val goal: String,
    val createdAt: String,
    val photoCount: Int,
    val totalSteps: Int,
    val totalDistanceM: Float,
    val dir: File,
)

class LocalSessionStore(private val context: Context) {

    private val rootDir: File
        get() = File(context.filesDir, "snav_local").also { it.mkdirs() }

    fun createSession(goal: String): String {
        val id = "local_${System.currentTimeMillis()}"
        val dir = File(rootDir, id).also {
            File(it, "photos").mkdirs()
            File(it, "sensors").mkdirs()
        }
        val manifest = JSONObject().apply {
            put("id", id)
            put("goal", goal)
            put("created_at", dateFormat().format(Date()))
            put("entries", JSONArray())
        }
        File(dir, "manifest.json").writeText(manifest.toString(2))
        return id
    }

    fun saveEntry(
        sessionId: String,
        photoFile: File,
        pdrSnapshot: PdrSnapshot?,
    ): Int {
        val dir = File(rootDir, sessionId)
        if (!dir.exists()) return -1

        val manifest = readManifest(dir)
        val entries = manifest.getJSONArray("entries")
        val index = entries.length() + 1

        val photoName = "photo_$index.jpg"
        photoFile.copyTo(File(dir, "photos/$photoName"), overwrite = true)

        if (pdrSnapshot != null) {
            val sensorName = "sensors_$index.json"
            val sensorJson = JSONObject().apply {
                put("pdr", JSONObject().apply {
                    put("segment_steps", pdrSnapshot.segmentSteps)
                    put("segment_distance_m", pdrSnapshot.segmentDistanceM)
                    put("avg_heading_deg", pdrSnapshot.avgHeadingDeg)
                    put("current_x", pdrSnapshot.currentX)
                    put("current_y", pdrSnapshot.currentY)
                    put("total_steps", pdrSnapshot.totalSteps)
                    put("total_distance_m", pdrSnapshot.totalDistanceM)
                })
                put("raw_sensors", pdrSnapshot.rawSensorJson)
            }
            File(dir, "sensors/$sensorName").writeText(sensorJson.toString())
        }

        entries.put(JSONObject().apply {
            put("index", index)
            put("photo", "photos/$photoName")
            put("sensor", if (pdrSnapshot != null) "sensors/sensors_$index.json" else JSONObject.NULL)
            put("timestamp", dateFormat().format(Date()))
            put("steps", pdrSnapshot?.totalSteps ?: 0)
            put("distance_m", pdrSnapshot?.totalDistanceM ?: 0f)
            put("pdr_x", pdrSnapshot?.currentX ?: 0f)
            put("pdr_y", pdrSnapshot?.currentY ?: 0f)
            put("heading_deg", pdrSnapshot?.avgHeadingDeg ?: 0f)
            put("segment_steps", pdrSnapshot?.segmentSteps ?: 0)
            put("segment_distance_m", pdrSnapshot?.segmentDistanceM ?: 0f)
            put("game_rot_vec_yaw_deg", pdrSnapshot?.gameRotVecYawDeg ?: 0f)
            put("rot_vec_yaw_deg", pdrSnapshot?.rotVecYawDeg ?: 0f)
            put("azimuth_deg", pdrSnapshot?.azimuthDeg ?: 0f)
            put("complementary_yaw_deg", pdrSnapshot?.complementaryYawDeg ?: 0f)
            put("calibrated_yaw_deg", pdrSnapshot?.calibratedYawDeg ?: 0f)
            put("gps_lat", pdrSnapshot?.gpsLat ?: JSONObject.NULL)
            put("gps_lng", pdrSnapshot?.gpsLng ?: JSONObject.NULL)
            put("gps_accuracy", pdrSnapshot?.gpsAccuracy ?: JSONObject.NULL)
            if (pdrSnapshot != null) {
                put("game_rot_vec_quat", JSONArray().apply {
                    pdrSnapshot.gameRotVecQuat.forEach { put(it.toDouble()) }
                })
                put("rot_vec_quat", JSONArray().apply {
                    pdrSnapshot.rotVecQuat.forEach { put(it.toDouble()) }
                })
            }
        })

        manifest.put("entries", entries)
        File(dir, "manifest.json").writeText(manifest.toString(2))
        return index
    }

    fun listSessions(): List<LocalSession> {
        return rootDir.listFiles()
            ?.filter { it.isDirectory && File(it, "manifest.json").exists() }
            ?.mapNotNull { dir ->
                try {
                    val m = readManifest(dir)
                    val entries = m.getJSONArray("entries")
                    val lastEntry = if (entries.length() > 0) entries.getJSONObject(entries.length() - 1) else null
                    LocalSession(
                        id = m.getString("id"),
                        goal = m.getString("goal"),
                        createdAt = m.getString("created_at"),
                        photoCount = entries.length(),
                        totalSteps = lastEntry?.optInt("steps", 0) ?: 0,
                        totalDistanceM = lastEntry?.optDouble("distance_m", 0.0)?.toFloat() ?: 0f,
                        dir = dir,
                    )
                } catch (e: Exception) {
                    Log.w("LocalStore", "Skip bad session dir: ${dir.name}", e)
                    null
                }
            }
            ?.sortedByDescending { it.createdAt }
            ?: emptyList()
    }

    fun savePdrPath(sessionId: String, pathPoints: List<PdrPoint>) {
        val dir = File(rootDir, sessionId)
        if (!dir.exists()) return
        val arr = JSONArray()
        for (p in pathPoints) {
            arr.put(JSONObject().apply {
                put("x", p.x.toDouble())
                put("y", p.y.toDouble())
                put("step", p.stepIndex)
            })
        }
        File(dir, "pdr_path.json").writeText(arr.toString())
    }

    fun loadPdrPath(sessionId: String): List<PdrPoint> {
        val file = File(rootDir, "$sessionId/pdr_path.json")
        if (!file.exists()) return emptyList()
        return try {
            val arr = JSONArray(file.readText())
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                PdrPoint(o.getDouble("x").toFloat(), o.getDouble("y").toFloat(), o.getInt("step"))
            }
        } catch (_: Exception) { emptyList() }
    }

    fun getSessionDir(sessionId: String): File? {
        val dir = File(rootDir, sessionId)
        return if (dir.exists()) dir else null
    }

    fun deleteSession(sessionId: String) {
        File(rootDir, sessionId).deleteRecursively()
    }

    fun exportSessionZip(sessionId: String): File? {
        val dir = File(rootDir, sessionId)
        if (!dir.exists()) return null

        val manifest = readManifest(dir)
        val goal = manifest.optString("goal", sessionId)
        val safeName = goal.replace(Regex("[^\\w\\u4e00-\\u9fff]"), "_").take(30)
        val zipFile = File(context.cacheDir, "snav_${safeName}.zip")

        ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
            addDirToZip(zos, dir, dir.name)
        }
        return zipFile
    }

    fun shareSession(sessionId: String): Intent? {
        val zipFile = exportSessionZip(sessionId) ?: return null
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            zipFile,
        )
        return Intent(Intent.ACTION_SEND).apply {
            type = "application/zip"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    private fun addDirToZip(zos: ZipOutputStream, file: File, basePath: String) {
        if (file.isDirectory) {
            file.listFiles()?.forEach { child ->
                addDirToZip(zos, child, "$basePath/${child.name}")
            }
        } else {
            zos.putNextEntry(ZipEntry(basePath))
            FileInputStream(file).use { it.copyTo(zos) }
            zos.closeEntry()
        }
    }

    private fun readManifest(dir: File): JSONObject {
        return JSONObject(File(dir, "manifest.json").readText())
    }

    private fun dateFormat() = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
}
