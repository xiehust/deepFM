# -*- coding: utf-8 -*-

from sklearn import preprocessing
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import nd, autograd, gluon

MODETYPE=0
TRAINFILE='./data/test.txt'
model_ctx = mx.gpu()
p_drop = 0
class Net(gluon.HybridBlock):
    def __init__(self, k_units, feature_dim, dnn_layers=[256, 128, 64], **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # for DNN part
            self.denselayers = []
            self.droplayers = []
            self._dnn_layers = dnn_layers

            self._k_units = k_units
            self._feature_dim = feature_dim
            for i in dnn_layers:
                self.denselayers.append(gluon.nn.Dense(i, activation='relu'))
                self.droplayers.append(gluon.nn.Dropout(p_drop))

            # Note that Blocks inside the list,
            # tuple or dict will not be registered automatically.
            # Make sure to register them using register_child()
            for i in range(len(self._dnn_layers)):
                self.register_child(self.denselayers[i])
                self.register_child(self.droplayers[i])

            self.densefinal = gluon.nn.Dense(1)

            # FM part 1st order layer
            self.w = self.params.get('w', grad_req='write', shape=(self._feature_dim, 1))

            # FM part 2nd order vector
            self.v = self.params.get('v', grad_req='write', shape=(self._feature_dim, k_units))

    def hybrid_forward(self, F, x,w, v):
        x_broadcast = x.reshape((-1, self._feature_dim, 1))
        # FM first order
        f1 = F.broadcast_mul(w, x_broadcast)
        f1 = F.sum(f1, axis=2)

        # FM 2nd order
        xv = F.broadcast_mul(v, x_broadcast)
        f2 = 0.5 * (F.square(F.sum(xv, axis=1)) - F.sum(F.square(xv), axis=1))
        f2 = F.Dropout(f2,p_drop)
        # DNN
        xv = F.flatten(xv) #shape=[-1,self._feature_dim*self._k_units])
        #print (xv.shape)
        for i in range(len(self._dnn_layers)):
            xv = self.denselayers[i](xv)
            xv = self.droplayers[i](xv)
        # concate all parts
        y = F.concat(xv, f1, f2, dim=1)
        y = self.densefinal(y)
        return y

def processData(fname):
    ##read file
    ids = [str(i) for i in range(0, 10)]
    feature = [str(i) for i in range(1, 10)]

    all_df = pd.read_table(fname, header=None, encoding='utf8', sep=',', skiprows=1, names=ids)
    all_df = all_df.sample(frac=1)

    feature_df = all_df[feature]
    scalafeature = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(feature_df)
    scalafeature_frame = pd.DataFrame(scalafeature, columns=feature_df.columns)

    y_data = np.array(all_df['0']).reshape(-1, 1)
    x_data = np.array(scalafeature_frame).reshape(-1, len(feature))
    batch_size = 128
    train_dataset = gluon.data.ArrayDataset(x_data.astype('float32'),
                                            y_data.astype('float32'))
    train_dataloader = mx.gluon.data.DataLoader(train_dataset, batch_size, shuffle=True)

    return train_dataloader,x_data.shape[0]

def logistic(z):
    return (1. / (1. + nd.exp(-z)))

def log_loss(output, y):
    yhat = logistic(output)
    return  - nd.sum(  y * nd.log(yhat) + (1-y) * nd.log(1-yhat))

def createNet(k_units,dim):
    return Net(k_units,dim)

def train(train_data,net,batch_size):
    print(net.collect_params())
    net.collect_params().initialize(mx.init.Normal(sigma=0.01),ctx=model_ctx,force_reinit=True)
    net.hybridize()
    # trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .2, 'wd': 0.001, 'momentum': 0.9})
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})
    # trainer = gluon.Trainer(net.collect_params(), 'rmsprop',
    #                             {'learning_rate': 0.01, 'gamma1': 0.9})
    epochs = 200
    loss_sequence = []
    failcnt = 0
    for epoch in range(epochs):
        cumulative_loss = 0
        t1 = time.time()
        for i, (a, label) in enumerate(train_data):
            a = a.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(a)
                loss = log_loss(output, label)
            loss.backward()
            params = net.collect_params()
            for param in params:
                print(param.grad)
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        print("Epoch %s, cumulative loss: %s,average loss:%.5f, time cost: %.2f" % (epoch,
                                                                                    cumulative_loss,
                                                                                    cumulative_loss / length,
                                                                                    time.time() - t1))
        loss_sequence.append(cumulative_loss)
        if epoch > 0:
            if loss_sequence[epoch - 1] < cumulative_loss:
                failcnt += 1
            else:
                failcnt = 0
        if failcnt > 2:
            break
    net.save_params('./tmp/mxmodel.params')
    return net

def predict_auc(net,dataloader,length):
    num_correct = 0.0
    num_total = length
    prediclist = []
    labellist = []

    for i, (a, label) in enumerate(dataloader):
        a = a.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = net(a)
        prediction = (nd.sign(output) + 1) / 2
        logi = logistic(output)
        prediclist.append(logi.asnumpy())
        labellist.append(label.asnumpy())
        num_correct += nd.sum(prediction == label)
    print("Accuracy: %0.3f (%s/%s)" % (num_correct.asscalar() / num_total, num_correct.asscalar(), num_total))

    from sklearn.metrics import roc_curve, auc
    ytrue = [item[0] for batch in labellist for item in batch]
    ypred = [item[0] for batch in prediclist for item in batch]
    fpr, tpr, thresholds = roc_curve(ytrue, ypred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (1, roc_auc))
    print ('AUC: %.5f' % roc_auc)

if __name__ == '__main__':
    k_units = 10  # latent feature for fm #
    batch_size = 128
    train_dataloader, length  = processData(TRAINFILE)
    net = createNet(k_units,dim=9)
    net = train(train_dataloader,net,batch_size)
    predict_auc(net,train_dataloader,length)


