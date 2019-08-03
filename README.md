# Cast Search by Portrait
WIDER Face & Person Challenge 2019 - Track 3: Cast Search by Portrait


## Environment & Dependency
- python3
- tqdm 
- numpy, skimage, opencv
- tensorboardX
- torch 1.1.0 torchvision
- mxnet 1.1.0

## Data preparation
1. Download the `train.json, val.json, test.json, val_label.json`, and put them in `data/` folder.
2. Download the original data `train.tar val.tar, test.tar`, and extract them in `$ORIGIN` folder.
3. Crop the body images into `$BODY` folder and save the body information in `data/body_train.pkl, data/body_val.pkl. data/body_test.pkl`.
```
python3 crop_body.py --origin $ORIGIN --body $BODY --mode train
python3 crop_body.py --origin $ORIGIN --body $BODY --mode val
python3 crop_body.py --origin $ORIGIN --body $BODY --mode test
```


## Face detection and face features
The following commands run the result on validation set, you can replace `--mode val` with `--mode test` to get test set result.

1. `cd RetinaFace`
2. Type ``make`` to build cxx tools.
3. Download pretrained model, put them in `model/`
    - face detection: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [dropbox](https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0))
    - face recognition: LResNet100E-IR,ArcFace@ms1m-refine-v2 ([baidu cloud](https://pan.baidu.com/s/1wuRTf2YIsKt76TxFufsRNA) or [dropbox](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0))
4. Generate file lists.
```
python3 datalist.py --root $BODY --mode val
```
5. Dectect face for every file list in `data/`, and save the result in `results/`
```
# ${FILE}.txt is the data list in data/
python3 deploy --txt_file data/${FILE}.txt --pkl_file results/${FILE}.pkl \
               --image_root $BODY
```
6. Merge the result into single file `../data/retina_val.pkl`.
```
python3 merge.py --mode val
```
7. Select the face bboxes and crop face.
```
python3 face_det.py --image_root $BODY --mode val
```
8. Extract face features and save it into `../data/face_val.pkl`.
```
python3 face_feat.py --mode val
```


## ReID model and body features
The following commands run the result on validation set, you can replace `--mode val` with `--mode test` to get test set result.

1. `cd reid`
2. Train a reid model with batch hard triplet loss or use MGN model. You can use `--use_val` to train the model with training set and validation set.
```
python3 triplet.py --image_root $BODY --cnn resnet50
python3 mgn.py --image_root $BODY --cnn resnet50 --use_val
```
We provide multiple cnn backbone selections.
 - `'resnet50', 'resnet101', 'resnet152'`
 - `'resnext50_32x4d', 'resnext101_32x8d'`
 - `'densenet121', 'densenet161', 'densenet169', 'densenet201'`

You can download our [pre-trained models](https://drive.google.com/drive/folders/11pf0GH51g0l2B2HhhRxa2welaGnhVK4y?usp=sharing). These models are trained with both training set and validation set.
 
3. Extract body features.
```
python3 triplet.py --image_root $BODY --cnn resnet50 --eval --mode val
python3 mgn.py --image_root $BODY --cnn resnet50 --eval --mode val
```
4. Move body features `triplet_resnet50_val.pkl, mgn_resnet50_val.pkl` to `../data/` folder.


## Final result
Run `fusion.py` to get the ranking result `val_result.txt`. 
```
python3 fusion.py --mode val
```
You can use `--face_rerank` or `--body_rerank` to do rerank on face distances matrix or body distances matrix respectively. These will improve the perfermence.


# Github projects

- face detection and face recognition from [insightface](https://github.com/deepinsight/insightface)
- random erasing from [reid_baseline](https://github.com/L1aoXingyu/reid_baseline)
- k-reciprocal rerank from [person-re-reranking](https://github.com/zhunzhong07/person-re-ranking)



# Reference 

[1] Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, and Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild." arXiv preprint arXiv:1905.00641, 2019.

[2] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. "Arcface: Additive angular margin loss for deep face recognition." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4690-4699. 2019.

[3] Alexander Hermans, Lucas Beyer, and Bastian Leibe. "In defense of the triplet loss for person re-identification." arXiv preprint arXiv:1703.07737, 2017.

[4] Guanshuo Wang, Yufeng Yuan, Xiong Chen, Jiwei Li, Xi Zhou. "Learning discriminative features with multiple granularities for person re-identification." 2018 ACM Multimedia Conference on Multimedia Conference. ACM, 2018.

[5] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. "Random erasing data augmentation." arXiv preprint arXiv:1708.04896, 2017.

[6] Zhun Zhong , Liang Zheng, Donglin Cao, and Shaozi Li. "Re-ranking person re-identification with k-reciprocal encoding." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1318-1327. 2017.
