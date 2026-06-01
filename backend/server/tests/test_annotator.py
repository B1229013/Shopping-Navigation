from PIL import Image
from server.annotator import annotate
from server.perception import Detection


def test_annotate_writes_png(tmp_path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (200, 200), color=(255, 255, 255)).save(src)
    dst = tmp_path / "out.jpg"
    detections = [Detection(label="cat", box=[10, 10, 80, 80], score=0.9)]
    annotate(str(src), str(dst), detections, banner_text="GUIDANCE: turn left")
    assert dst.exists()
    out = Image.open(dst)
    assert out.size[0] == 200 and out.size[1] >= 200  # banner adds height
