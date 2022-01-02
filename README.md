# pseudo_label_contrastive_training
Code for Pseudo label based contrastive learning joint training approach

**Local contrastive loss with pseudo-label based self-training for semi-supervised medical image segmentation** <br/>

The code is for the article "Local contrastive loss with pseudo-label based self-training for semi-supervised medical image segmentation" under review. With the proposed joint-training method using Contrastive loss, we get competitive segmentation performance with just 2 labeled training volumes compared to upperbound and compared methods.<br/>
https://arxiv.org/abs/2112.09645 <br/>

**Authors:** <br/>
Krishna Chaitanya ([email](mailto:krishna.chaitanya@vision.ee.ethz.ch)),<br/>
Ertunc Erdil,<br/>
Neerav Karani,<br/>
Ender Konukoglu.<br/>

**Requirements:** <br/>
Python 3.6.1,<br/>
Tensorflow 1.12.0,<br/>
rest of the requirements are mentioned in the "requirements.txt" file. <br/>

I)  To clone the git repository.<br/>
git clone https://github.com/krishnabits001/pseudo_label_contrastive_training.git <br/>

II) Install python, required packages and tensorflow.<br/>
Then, install python packages required using below command or the packages mentioned in the file.<br/>
pip install -r requirements.txt <br/>

To install tensorflow <br/>
pip install tensorflow-gpu=1.12.0 <br/>

III) Dataset download.<br/>
To download the ACDC Cardiac dataset, check the website :<br/>
https://www.creatis.insa-lyon.fr/Challenge/acdc. <br/>

To download the Medical Decathlon Prostate dataset, check the website :<br/>
http://medicaldecathlon.com/

To download the MMWHS Cardiac dataset, check the website :<br/>
http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/
 
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. For more details, refer to the "N4_bias_correction.py" file in scripts.<br/>
Image and label pairs are re-sampled (to chosen target resolution) and cropped/zero-padded to a fixed size using "create_cropped_imgs.py" file. <br/>

IV) Train the model.<br/>
To do joint training run the script "pseudo_lbl_rand_init.sh" in train_model directory.<br/>
For instance, if we want to train for ACDC dataset with 2 training volumes and configuration c1 use below step.<br/>
bash pseudo_lbl_rand_init.sh tr2 c1 acdc

<br/>
Above command, executes the below 2 steps of training: <br/>
Steps :<br/>
1) In Step 1: Train a baseline network model to infer the initial pseudo-labels for unlabeled data. This training is only done once at the start. <br/> 
cd train_model/ <br/>
python tr_baseline.py --no_of_tr_imgs=tr2 --comb_tr_imgs=c1 --dataset=acdc

2) In Step 2: Post Step 1, we infer pseudo-labels of unlabeled data and perform the joint training based on contrastive loss and segmentation loss. This training is done iteratively, where the pseudo-labels are refined periodicallt.<br/>
python prop_method_joint_tr_rand_init.py --no_of_tr_imgs=tr2 --comb_tr_imgs=c1 --dataset=acdc 

V) Config files contents.<br/>
One can modify the contents of the below 2 config files to run the required experiments.<br/>
experiment_init directory contains 2 files.<br/>
Example for ACDC dataset:<br/>
1) init_acdc.py <br/>
--> contains the config details like target resolution, image dimensions, data path where the dataset is stored and path to save the trained models.<br/>
2) data_cfg_acdc.py <br/>
--> contains an example of data config details where one can set the patient ids which they want to use as train, validation and test images.<br/>


**Bibtex citation:** 

@article{chaitanya2021local,
  title={Local contrastive loss with pseudo-label based self-training for semi-supervised medical image segmentation},
  author={Chaitanya, Krishna and Erdil, Ertunc and Karani, Neerav and Konukoglu, Ender},
  journal={arXiv preprint arXiv:2112.09645},
  year={2021}
}
