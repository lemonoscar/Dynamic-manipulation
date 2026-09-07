import base64,io
import pytest
from PIL import Image
from scripts.run_sim6_rtc_pick import compare_first_rgb

def encoded(size,color):
    b=io.BytesIO();Image.new('RGB',size,color).save(b,format='PNG');return base64.b64encode(b.getvalue()).decode()

def test_independent_rgb_is_measured_and_wrong_dimensions_rejected():
    a=encoded((4,3),(1,2,3));b=encoded((4,3),(2,3,4))
    old={'head_images':[a,a],'wrist_images':[a,a]};new={'head_images':[b,b],'wrist_images':[b,b]}
    result=compare_first_rgb(old,new)
    assert all(row['pixel_mae_255']==1 and not row['jpeg_sha_equal'] for values in result.values() for row in values)
    new['head_images'][0]=encoded((5,3),(2,3,4))
    with pytest.raises(ValueError,match='dimensions'):compare_first_rgb(old,new)
