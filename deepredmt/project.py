import tensorflow as tf
import umap.umap_ as umap
import matplotlib.pyplot as plt

from . import data_handler

def project(raw_data, tf_model):

    # read windows
    x, y = data_handler.read_windows(
        raw_data,
        read_labels=True,
        read_edexts=False,
        occlude_target=False)
    # x = data_handler.read_windows(
    #     raw_data,
    #     read_labels=False,
    #     read_edexts=False,
    #     occlude_target=False)[0]

    m = tf.keras.models.load_model(tf_model,
                                   compile=False)

    codes = m.encoder.predict(tf.one_hot(x, depth=4), batch_size=512)

    reducer = umap.UMAP(random_state=42)
    proj = reducer.fit_transform(codes)

    colors=['tab:blue' if i == 0 else 'black' for i in y]
    f = plt.figure()
    plt.rcParams["figure.figsize"] = (10,10)
    plt.scatter(proj[:,0], proj[:,1], c=colors)
    # plt.scatter(proj[:,0], proj[:,1])
    plt.show()
#    f.savefig(outfname, bbox_inches='tight')

    return None
