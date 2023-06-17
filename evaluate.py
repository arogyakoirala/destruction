from keras.models import load_model
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tensorflow.keras import metrics
from tensorflow.keras.utils import Sequence
import numpy as np
import zarr


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("run_id", help="One of snn, double")
parser.add_argument("--output_dir", help="Output dir")
parser.add_argument("--data_dir", help="Path to data dir")
args = parser.parse_args()


# For local
OUTPUT_DIR = "../data/destr_outputs"
DATA_DIR = "../data/destr_data"

## For artemisa
OUTPUT_DIR = "/lustre/ific.uv.es/ml/iae091/outputs"
DATA_DIR = "/lustre/ific.uv.es/ml/iae091/data"

BATCH_SIZE = 32

if args.output_dir:
    OUTPUT_DIR = args.output_dir

if args.data_dir:
    DATA_DIR = args.data_dir

RUN_DIR = OUTPUT_DIR + f"/{args.run_id}"
MODEL_STORAGE_LOCATION = f"{RUN_DIR}/model"

fp = open(f"{RUN_DIR}/metadata.txt")
for i, line in enumerate(fp):
    if i == 2:
        CITIES = line.split("[")[-1].split("]")[0].replace("'", "").split(", ")
    if i > 2:
        break
fp.close()

print(f"Identifed cities: {CITIES}")

TRAINING_DATA_DIR = OUTPUT_DIR + f"/data/{'-'.join(CITIES)}"


class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=BATCH_SIZE, train=True):
        self.images_pre = images[0]
        self.images_post = images[1]
        self.labels = labels
        self.batch_size = batch_size
        self.train = train


        
        # self.tuple_pairs = make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        # np.random.shuffle(self.tuple_pairs)
    def __len__(self):
        return len(self.images_pre)//self.batch_size    
    
    def __getitem__(self, index):
        X_pre = self.images_pre[index*self.batch_size:(index+1)*self.batch_size].astype('float') / 255.0
        X_post = self.images_post[index*self.batch_size:(index+1)*self.batch_size].astype('float') / 255.0
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        if self.train:
            return {'images_t0': X_pre, 'images_tt': X_post}, y
        else:
            return {'images_t0': X_pre, 'images_tt': X_post}
            

im_te_pre = zarr.open(f"{TRAINING_DATA_DIR}/im_te_pre.zarr")
im_te_post = zarr.open(f"{TRAINING_DATA_DIR}/im_te_post.zarr")
la_te = zarr.open(f"{TRAINING_DATA_DIR}/la_te.zarr")


best_model = load_model(MODEL_STORAGE_LOCATION, custom_objects={'auc':metrics.AUC(num_thresholds=200, curve='ROC', name='auc')})
gen_te= SiameseGenerator((im_te_pre, im_te_post), la_te, train=False)
yhat_proba, y = np.squeeze(best_model.predict(gen_te)), np.squeeze(la_te[0:(len(la_te)//BATCH_SIZE)*BATCH_SIZE])
roc_auc_test = roc_auc_score(y, yhat_proba)
# #calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y, yhat_proba)


# #create precision recall curve
# fig, ax = plt.subplots()
# ax.plot(recall, precision, color='purple')

# #add axis labels to plot
# ax.set_title('Precision-Recall Curve')
# ax.set_ylabel('Precision')
# ax.set_xlabel('Recall')
# f = open(f"{RUN_DIR}/metadata.txt", "a")
# f.write("\n\n######## Test set performance\n\n")
# f.write(f'Test Set AUC Score for the ROC Curve: {roc_auc_test} \nAverage precision:  {np.mean(precision)}')
print(f"""
    Test Set AUC Score for the ROC Curve: {roc_auc_test} 
    Average precision:  {np.mean(precision)}
""")
# f.close()