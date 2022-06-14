# Towards Attack to AttriGuard Through Some Attempts
We employ 0-norm projected gradient descent (PGD) for adversarial training (AT), while using simple random sampling and early stopping to train a more robust attack model for attacking [AttriGuard](https://arxiv.org/abs/1805.04810). We conduct experiments using the AttriGuard [dataset](https://github.com/jjy1994/AttriGuard), and the experimental results demonstrate the effectiveness of the attack.
## Experimental result
<p>
  <img src="https://github.com/gxx1506215897/attack_AttriGuard/blob/master/experimental_results/att.png" alt='images' width="500"/>
</p>

We used 9 attack methods to attack AttriGuard. The attack structure is shown in Fig. In the process of increasing the data loss budget, the inference accuracy of our attack decreases more slowly relative to the attack in AttriGuard. The experimental results show that our attack is effective.

| City | AUC         | Accuracy    | City | AUC         | Accuracy    |
|------|-------------|-------------|------|-------------|-------------|
| 0    | 0.892512449 | 0.756028369 | 13   | 0.840914441 | 0.695035461 |
| 1    | 0.891177003 | 0.734751773 | 14   | 0.936038932 | 0.84964539  |
| 2    | 0.765803531 | 0.663829787 | 15   | 0.927926664 | 0.84964539  |
| 3    | 0.897365324 | 0.780141844 | 16   | 0.944178361 | 0.84822695  |
| 4    | 0.832793119 | 0.70070922  | 17   | 0.882231779 | 0.756028369 |
| 5    | 0.919366229 | 0.763120567 | 18   | 0.930656406 | 0.825531915 |
| 6    | 0.839791761 | 0.74893617  | 19   | 0.953245813 | 0.873758865 |
| 7    | 0.868424627 | 0.771631206 | 20   | 0.917990041 | 0.797163121 |
| 8    | 0.946523314 | 0.84964539  | 21   | 0.959049344 | 0.878014184 |
| 9    | 0.786084201 | 0.70212766  | 22   | 0.919416025 | 0.812765957 |
| 10   | 0.854454504 | 0.714893617 | 23   | 0.953037574 | 0.879432624 |
| 11   | 0.867913083 | 0.737588652 | 24   | 0.939289271 | 0.826950355 |
| 12   | 0.895622454 | 0.782978723 |      |             |             |

The results of LID detection is shown in the Table. We can see that the detection accuracy is high, and the AUC, the area under the curve, is also high. The highest accuracy rate was 0.88, the lowest accuracy rate was 0.66, and the median was 0.78. The highest AUC value was 0.96, the lowest AUC value was 0.77, and the median was 0.90.We can see that the detection works well

## Code usage
### More Robust Attack Model
* The folder att_data is used to store AttriGuard data.
* models are used to store the trained models.
* att_NiN_bn.py defines the structure of the attack model.
* config.json defines many parameters whose values can be changed.
* input_data.py is used to import the AttriGuard dataset.
* pgd_attack.py contains our 0-norm PGD method.
* train_att_model.py trains a more robust attack model, using 0-norm PGD and simple random sampling.
* generate_validaion.py is used to split the original training set into new training and validation sets.
* train_att_model_new.py is the training code of the attack model after adding early stopping.
* The parameters stored in config_new.json are the parameters used in train_att_model_new.py.
#### The following code can be used to train the attack model:
```
python train_att_model.py
```
### LID detector
* The corresponding folder is 'lid_detector/'. The following files or folders are in that folder.
* The folder att_data is the dataset of the AttriGuard.
* The folder cleverhans is the python library we used when producing the adversarial examples.
* The folder data is to store the adversarial examples and noisy examples.
* The folder data_grid_search is used to store the characteristics of the data.
* attacks.py is used to store the methods to craft the adversarial examples.
* cw_attacks.py is used to store the CW method to craft the adversarial examples.
* util.py is used to store some useful functions when doing the detection.
* craft_adv_examples.py is used to craft the adversarial examples.
* extract_characteristics.py is used to get the LID value of the adversarial examples.
* extract_characteristics1.py is used to get the LID value of the adversarial examples produced by AttriGuard.
* detect_adv_examples1.py is used to detect the adversarial examples of the AttriGuard, and it will give the results of the detector.
#### The following code can be used to do the detection:
##### Step 1: train the target attack model.
```
python train_model.py
```
##### Step 2: craft adversarial examples.
```
python craft_adv_examples.py -a jsma -b 100
```
##### Step 3: extract characteristics from adversarial examples produced by JSMA.
```
python extract_characteristics.py -d att -a jsma -r lid -k 20 -b 100
```
##### Step 4: extract characteristics from adversarial examples produced by AttriGuard.
```
python extract_characteristics1.py -d att -a jsma -r lid -k 20 -b 100
```
##### Step 5: detect the adversarial examples produced by AttriGuard.
```
python detect_adv_examples1.py -d att -a jsma -r lid -t attriguard -b 100
```
* The result is stored in 'tongji2.csv'. 
* 'tongji.csv' is used to store the results of the FGSM. 'tongji2.csv' is used to store the results of the BIM.
If you have any question, please feel free to send email to guangxxie@126.com.
## Citation
If you use this code or dataset, please cite following paper:
```
@inproceedings{Xie2022,
  title={Towards Attack to AttriGuard Through Some Attempts},
  author={Guangxu Xie, Qingqi Pei, Yunwei Wang, and Gaopan Hou},
  booktitle={Security and Communication Networks},
  year={2022}
}
```
