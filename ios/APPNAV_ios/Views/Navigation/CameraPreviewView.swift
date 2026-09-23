import AVFoundation
import Combine
import SwiftUI
import UIKit

/// Owns the capture session and exposes a simple capture-photo API; mirrors the
/// role of Android's `imageCapture: ImageCapture` in `NavigationViewModel.kt`.
final class CameraController: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate {
    let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var captureCompletion: ((Data?) -> Void)?
    private var videoDevice: AVCaptureDevice?

    @Published private(set) var isTorchAvailable = false
    @Published private(set) var isTorchOn = false

    func configure() {
        sessionQueue.async {
            self.session.beginConfiguration()
            self.session.sessionPreset = .photo

            if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
               let input = try? AVCaptureDeviceInput(device: device),
               self.session.canAddInput(input) {
                self.session.addInput(input)
                self.videoDevice = device
                let hasTorch = device.hasTorch
                DispatchQueue.main.async { self.isTorchAvailable = hasTorch }
            }
            if self.session.canAddOutput(self.photoOutput) {
                self.session.addOutput(self.photoOutput)
            }

            self.session.commitConfiguration()
        }
    }

    func start() {
        sessionQueue.async {
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
            self.setTorch(on: false)
        }
    }

    /// Toggles the continuous torch (flashlight) used to light the scene before capture,
    /// since low-light store aisles otherwise leave the VLM guidance photo too dark to read.
    func toggleTorch() {
        sessionQueue.async {
            guard let device = self.videoDevice else { return }
            self.setTorch(on: device.torchMode != .on)
        }
    }

    /// Must be called on `sessionQueue`.
    private func setTorch(on: Bool) {
        guard let device = videoDevice, device.hasTorch else { return }
        do {
            try device.lockForConfiguration()
            if on {
                try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
            } else {
                device.torchMode = .off
            }
            device.unlockForConfiguration()
            DispatchQueue.main.async { self.isTorchOn = on }
        } catch {
            DispatchQueue.main.async { self.isTorchOn = device.torchMode == .on }
        }
    }

    func capturePhoto(completion: @escaping (Data?) -> Void) {
        sessionQueue.async {
            self.captureCompletion = completion
            let settings = AVCapturePhotoSettings()
            self.photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let data = error == nil ? photo.fileDataRepresentation() : nil
        DispatchQueue.main.async {
            self.captureCompletion?(data)
            self.captureCompletion = nil
        }
    }
}

/// Full-screen live camera preview backed by `CameraController.session`.
struct CameraPreviewView: UIViewRepresentable {
    @ObservedObject var controller: CameraController

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = controller.session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}
