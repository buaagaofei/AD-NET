from keras.models import *
from keras.layers import *
import keras.backend as K

import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.callbacks import ModelCheckpoint,TensorBoard
import os
from save_data import *
import os



def brain_CNN_interaction(input_size=(91,109,91,1)):
    img_input = Input(input_size, name='img_input')
    conv1 = block(8, img_input)
    #conv2 = block(16, conv1)
    conv3 = block(32, conv1)
    conv4 = block(64, conv3)
    #conv5 = block(128, conv4)
    #full = KL.GlobalAveragePooling3D()(conv5)
    #conv5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)
    full = Flatten()(conv4)

    full = KL.Dropout(0.25)(full)

    age_output = Dense(1, name='age_output')(full)
    age_input = Input((1,), name='age_input')
    r_input = Input((1,), name='r_input')
    m_input = Input((1,), name='m_input')
    age_diff = KL.Subtract(name='age_diff')([age_output, age_input])
    age_diff = KL.Lambda(lambda x: x+32.6,name='nor_to_zero')(age_diff)
    # m the mean gap
    # m = 5
    # 1/(2m) * max(-m,min(age_diff,m))


    negative_m = KL.Lambda(lambda x: -1.0*x,name='negative_m')(m_input)
    age_diff = KL.Lambda(lambda x: -1.0*x,name='age_diff_2')(age_diff)
    inter_min = KL.Maximum(name='inter_min_1')([age_diff, negative_m])

    inter_min = KL.Lambda(lambda x: -1.0*x,name='inter_min_2')(inter_min)


    inter_max = KL.Maximum(name='inter_max')([inter_min, negative_m])

    # inter_min = KL.Minimum(name='inter_min')([age_diff,m_input])


    #     m = KL.constant(value=5.0, dtype='float32')
    #     negative_m = K.constant(value=-5.0, dtype='float32')

    #     inter_min = KL.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([age_diff,m])
    #     inter_min = KL.Lambda(lambda x: K.min(x, axis=-1))(inter_min)

    #     inter_max = KL.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([inter_min,negative_m])
    #     inter_max = KL.Lambda(lambda x: K.max(x, axis=-1))(inter_max)


    # inter_min = KL.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([age_diff,m])
    # inter_min = KL.Lambda(lambda x: K.min(x, axis=-1))(inter_min)
    # inter_min = K.maximum(age_diff,m)



    #
    # inter_max = KL.Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([inter_min,negative_m])
    # inter_max = KL.Lambda(lambda x: K.max(x, axis=-1))(inter_max)
    #


    #     inter_max = K.maximum(inter_min,negative_m)

    w = KL.Lambda(lambda x: x[0] / (2*x[1]),name='w')([inter_max,m_input])

    wr = KL.Lambda(lambda x: x[0] * x[1], name='wr')([w,r_input])

    # single output
    diag_output = Dense(1, activation='sigmoid', name='diag_output')(full)
    neg_output = KL.Lambda(lambda x: 1.0 - x, name='neg_output')(diag_output)

    # t1 = KL.Lambda(lambda x: 0.5 + x)(wr)
    # t2 = KL.Lambda(lambda x: 0.5 - x)(wr)
    #
    # m1 = KL.Multiply(name='m1')([diag_output, t1])
    # m2 = KL.Multiply()([neg_output, t2])

    ad_output = KL.Lambda(lambda x: (0.5 + x[2]) * x[0] / ((0.5 + x[2]) * x[0] + (0.5 - x[2]) * x[1]), name='ad_output')(
        [diag_output, neg_output, wr])

    #ad_output = Dense(1, name='ad_output')(ad_output)

    model = Model(inputs=[img_input, age_input, m_input, r_input], outputs=[age_output, ad_output])
    #   model = Model(inputs=[img_input,age_input], outputs=age_output)

    return model


def block(channels, myinput,trainable=False):
    conv = Conv3D(channels, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv'+str(channels)+'_1')(myinput)
    conv.trainable = trainable
    conv = KL.Dropout(0.5,name=str(channels)+'dropout')(conv)
    conv.trainable = trainable
    conv = Conv3D(channels, 3, activation='relu', padding='same', kernel_initializer='he_normal',name='conv'+str(channels)+'_2')(conv)
    conv.trainable = trainable
    conv = BatchNormalization(name=str(channels)+'batch_nor')(conv)
    # conv = Activation.relu(conv)
    conv = MaxPooling3D(pool_size=(2, 2, 2),name=str(channels)+'max_pool')(conv)
    return conv



def train_model(train_img,train_age,train_diag,test_img,test_age,test_diag,paras,model_path=None):

    m, r, agew, adw, pfix = paras['m'],paras['r'],paras['agew'],paras['adw'],paras['pfix']

    ms = np.ones(train_diag.shape) * m
    rs = np.ones(train_diag.shape) * r


    model = brain_CNN_interaction()
    if model_path is not None:
        model.load_weights(model_path)
    adam = optimizers.Adam(lr=0.001,decay=0.005)

    loss_weights_map = {'age_output': agew, 'ad_output': adw}

    loss_map = {'age_output': 'mean_absolute_error', 'ad_output': 'binary_crossentropy'}

    metric_map = {'age_output': 'mean_squared_error', 'ad_output': 'accuracy'}
    #
    # loss_map = {'age_output': 'mean_absolute_error', 'ad_output': 'binary_crossentropy'}
    #
    # metric_map = {'age_output': 'mean_squared_error', 'ad_output': 'accuracy'}

    #metric_map = {'ad_output': 'accuracy'}

    model.compile(optimizer=adam, loss=loss_map, loss_weights=loss_weights_map, metrics=metric_map)

    input_map = {'img_input': train_img, 'age_input': train_age, 'm_input': ms, 'r_input': rs}
    output_map = {'age_output': train_age, 'ad_output': train_diag}

    # if test_img is None or test_age is None or test_diag is None:
    model.fit(input_map, output_map, epochs=paras['epochs'], batch_size=16, verbose=2, callbacks=callbacks_list,
              validation_split=0.1)

    model.save_weights('logs//'+pfix+'//final_weights.h5')


if __name__=='__main__':


    # Stage 1 train the model for age prediction only
    train_img,train_age = get_combined_data_MCI(['IXI','healthy'],'dataset')
    train_diag = np.ones(train_age.shape)

    train_img2,train_age2 = get_combined_data_MCI(['Healthy','NL'],'dataset')
    train_diag2 = np.ones(train_age2.shape)


    train_img = np.concatenate((train_img,train_img2),axis=0)
    train_age = np.concatenate((train_age,train_age2),axis=0)
    train_diag = np.concatenate((train_diag,train_diag2),axis=0)

    del train_img2

    np.random.seed(0)
    b = np.random.permutation(train_img.shape[0])
    train_img = train_img[b]
    train_age = train_age[b]
    train_diag = train_diag[b]

    m = 5.0
    r = 0.0
    agew = 1.0
    adw = 0.0

    paras = {}
    paras['m'] = 5.0
    paras['r'] = 0.0
    paras['agew'] = 1.0
    paras['adw'] = 0.0
    paras['epochs'] = 200
    paras['pfix'] = 'Predict_age_only_decay_0-005_MAE_no16_moredata_2'
    train_model(train_img, train_age, train_diag, None, None, None, paras)
