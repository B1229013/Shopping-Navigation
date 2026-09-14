import CoreLocation
import Foundation

struct GPSSample: Codable {
    var timestamp: Double
    var latitude: Double
    var longitude: Double
    var altitude: Double
    var horizontalAccuracy: Double
    var speed: Double
    var course: Double
}

/// A fully independent GPS logger — its own `CLLocationManager`, no reference to
/// `MotionPathTracker`/`SensorComboTracker` or anything else in this app. Exists purely so
/// there's a ground-truth track that can be recorded *simultaneously* with any sensor test
/// without the two being able to interfere with each other, per the user's explicit
/// request — this is deliberately not wired into the other trackers in any way.
final class GPSLogger: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private(set) var isTracking = false

    var onLocation: ((GPSSample) -> Void)?
    var onAuthorizationDenied: (() -> Void)?

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = kCLDistanceFilterNone
    }

    func start() {
        guard !isTracking else { return }
        switch manager.authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            onAuthorizationDenied?()
            return
        default:
            break
        }
        isTracking = true
        manager.startUpdatingLocation()
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        manager.stopUpdatingLocation()
    }

    func locationManagerDidChangeAuthorization(_ mgr: CLLocationManager) {
        guard isTracking else { return }
        if mgr.authorizationStatus == .authorizedWhenInUse || mgr.authorizationStatus == .authorizedAlways {
            mgr.startUpdatingLocation()
        } else if mgr.authorizationStatus == .denied || mgr.authorizationStatus == .restricted {
            isTracking = false
            onAuthorizationDenied?()
        }
    }

    func locationManager(_ mgr: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        for location in locations {
            onLocation?(GPSSample(
                timestamp: location.timestamp.timeIntervalSince1970 * 1000,
                latitude: location.coordinate.latitude,
                longitude: location.coordinate.longitude,
                altitude: location.altitude,
                horizontalAccuracy: location.horizontalAccuracy,
                speed: max(location.speed, 0),
                course: location.course
            ))
        }
    }
}
