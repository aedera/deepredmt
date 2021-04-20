import tensorflow as tf

import matplotlib.pyplot as plt

from . import data_handler

def project(raw_data, tf_model):
    import umap.umap_ as umap

    # read windows
    x, y, p = data_handler.read_windows(
        raw_data,
        read_labels=True,
        read_edexts=True,
        occlude_target=True)
    # x = data_handler.read_windows(
    #     raw_data,
    #     read_labels=False,
    #     read_edexts=False,
    #     occlude_target=False)[0]

    x = x[y == 1, :]
    p = p[y == 1]

    m = tf.keras.models.load_model(tf_model,
                                   compile=False)

    reducer = tf.keras.Model(inputs=m.input,
                             outputs=m.get_layer('dense').output)

    codes = reducer.predict(tf.one_hot(x, depth=4), batch_size=512)
    #breakpoint()
    reducer = umap.UMAP(random_state=42)
    codes = reducer.fit_transform(codes)

    # colors=['tab:blue' if i == 0 else 'black' for i in y]
    f = plt.figure()
    plt.rcParams["figure.figsize"] = (10,10)
    plt.scatter(codes[:,0], codes[:,1], c=p)
    # plt.scatter(proj[:,0], proj[:,1])
    plt.show()
#    f.savefig(outfname, bbox_inches='tight')

    return None
