# Detect-and-Save-Eyes-and-Mouth-by-RetinaFace

## Explain about this Repository
1. Retinaface를 이용해 얼굴 전체를 나타내는 bounding box와 5개의 landmark(눈에 2개, 코에 1개, 입에 2개)를 찾음.
2. Gamma Correction → Brightness Normalization
3. Contrast Normalization
4. 왼쪽 눈, 오른쪽 눈, 입 부분의 위치를 고정 → 얼굴이 옆으로 기울어져도 바운딩 박스에서 눈 또는 입이 벗어나지 않게 하기 위함
5. 왼쪽 눈, 오른쪽 눈, 입 부분을 직사각형으로 crop함.
6. 프레임 별 .jpg 파일 저장 


## What's Difference?
  - Retinaface
  - Gamma correction
  - contrast normalizaiton


## Reference

### Retinaface
- paper: https://arxiv.org/abs/1905.00641

- github:

  official: https://github.com/deepinsight/insightface

  what I used: https://github.com/biubug6/Pytorch_Retinaface

### Gamma Correction
- gamma correction을 사용하여 drowsiness detection을 한 논문: https://www.mdpi.com/1999-5903/11/5/115

  > we applied gamma correction to enhance image contrast, which can help our method achieve better results,
  > 
  > noted by an increased accuracy of 2% in night environments.

- but, 논문에서 gamma 값을 명시하지 않아서 얼마로 할지 근거가 필요함.


### face normalization and recognition
- http://www1.cs.columbia.edu/~jebara/htmlpapers/UTHESIS/node46.html

- __face normalization이 얼굴 인식 분야에선 필요함.  but, face detection에선 꼭 필요할까? 에 대한 survey__

  paper: https://ieeexplore.ieee.org/document/5395980

  keypoint:

  >  Aim of the face detection system is to identify and locate all faces 
  >  
  >  regardless of their positions, scale, orientation, lighting conditions, expressions, etc

  > The recognition rate of 94 percent is obtained for original images whereas it increased to 97 percent for normalized images.

  _어쩌면 필요하지 않을지도.?_

### face alignment
- https://github.com/1adrianb/face-alignment

- what is blazeface?


### Face Image Normalization For Authentication Under Diverse Lighting Conditions 논문 참고

https://www.tdcommons.org/cgi/viewcontent.cgi?article=4488&context=dpubs_series
