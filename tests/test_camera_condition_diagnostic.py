from PIL import Image
import pytest
from scripts.diagnose_mani_conditioning import camera_variants


def test_camera_interventions_preserve_temporal_pairs_and_originals():
    rgb=[Image.new('RGB',(8,6),(i,0,0)) for i in range(1,5)]
    donor=[Image.new('RGB',(8,6),(i,0,0)) for i in range(5,9)]
    before=[im.tobytes() for im in rgb+donor]
    expected={'baseline':[1,2,3,4],'black_front':[0,0,3,4], 'black_wrist':[1,2,0,0],
        'swap_front':[5,6,3,4], 'swap_wrist':[1,2,7,8], 'black_rgb':[0,0,0,0],
        'exchange_camera_roles':[3,4,1,2]}
    for mode,frames in camera_variants(rgb,donor).items():
        assert [im.getpixel((0,0))[0] for im in frames]==expected[mode]
        assert all(im.size==(8,6) for im in frames)
    assert [im.tobytes() for im in rgb+donor]==before
    with pytest.raises(ValueError):camera_variants(rgb[:3],donor)
