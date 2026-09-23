import CoreLocation
import HealthKit

/// Pulls Apple Watch workout routes (GPS) out of HealthKit as a fallback source of
/// movement-path data — separate from CoreMotion, only useful when a workout was
/// actually started on the Watch and GPS had a fix (i.e. outdoors). Indoor workouts,
/// or no workout at all, have no route to recover; see WorkoutImportResult.hasRoute.
final class HealthKitService {
    static let shared = HealthKitService()
    private let store = HKHealthStore()

    var isAvailable: Bool { HKHealthStore.isHealthDataAvailable() }

    func requestAuthorization() async throws {
        guard isAvailable else { throw HealthKitServiceError.notAvailable }
        let workoutType = HKObjectType.workoutType()
        let routeType = HKSeriesType.workoutRoute()
        try await store.requestAuthorization(toShare: [], read: [workoutType, routeType])
    }

    func fetchRecentWorkouts(daysBack: Int = 3) async throws -> [HKWorkout] {
        let start = Calendar.current.date(byAdding: .day, value: -daysBack, to: Date()) ?? Date()
        let predicate = HKQuery.predicateForSamples(withStart: start, end: Date(), options: .strictStartDate)
        let sort = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)

        return try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: HKObjectType.workoutType(),
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [sort]
            ) { _, samples, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: (samples as? [HKWorkout]) ?? [])
            }
            store.execute(query)
        }
    }

    /// Empty array means the workout has no recorded GPS route (indoor workout, or no
    /// GPS fix at the time) — not an error, just nothing to import.
    func fetchRoute(for workout: HKWorkout) async throws -> [CLLocation] {
        let predicate = HKQuery.predicateForObjects(from: workout)
        let routes: [HKWorkoutRoute] = try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: HKSeriesType.workoutRoute(),
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { _, samples, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: (samples as? [HKWorkoutRoute]) ?? [])
            }
            store.execute(query)
        }
        guard let route = routes.first else { return [] }

        var allLocations: [CLLocation] = []
        return try await withCheckedThrowingContinuation { continuation in
            let routeQuery = HKWorkoutRouteQuery(route: route) { _, locations, done, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                if let locations {
                    allLocations.append(contentsOf: locations)
                }
                if done {
                    continuation.resume(returning: allLocations)
                }
            }
            store.execute(routeQuery)
        }
    }

    /// Projects GPS lat/lon to local meters (equirectangular, relative to the first
    /// fix) so it lines up with the same x/y convention MotionPathTracker uses.
    func convertToLocalPoints(_ locations: [CLLocation]) -> [PathPoint] {
        guard let origin = locations.first else { return [] }
        let earthRadius = 6_371_000.0
        let originLatRadians = origin.coordinate.latitude * .pi / 180

        return locations.map { location in
            let dLat = (location.coordinate.latitude - origin.coordinate.latitude) * .pi / 180
            let dLon = (location.coordinate.longitude - origin.coordinate.longitude) * .pi / 180
            let y = dLat * earthRadius
            let x = dLon * earthRadius * cos(originLatRadians)
            return PathPoint(
                x: x,
                y: y,
                headingDegrees: location.course >= 0 ? location.course : 0,
                timestamp: location.timestamp.timeIntervalSince1970 * 1000,
                stepCount: 0,
                distanceMeters: location.distance(from: origin)
            )
        }
    }

    enum HealthKitServiceError: Error {
        case notAvailable
    }
}

extension HKWorkoutActivityType {
    var displayName: String {
        switch self {
        case .walking: return "健走"
        case .running: return "跑步"
        case .hiking: return "健行"
        case .cycling: return "騎車"
        default: return "訓練"
        }
    }
}
