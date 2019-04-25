import keras

def unet(input_size=(572,572,1), pretrained_weights=None, print_summary=False):
    """ == Input === """
    inputs = keras.layers.Input(shape=input_size, name="input")

    """ === contracting path (follows the typical architecture of a convolutional network.) === """
    #=== 572×572×1 ===
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv1_1")(inputs)
    conv1 = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv1_2")(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    #=== 284×284×64 ===
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv2_1")(pool1)
    conv2 = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv2_2")(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    #=== 140×140×128 ===
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv3_1")(pool2)
    conv3 = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv3_2")(conv3)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    #=== 68×68×256 ===
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv4_1")(pool3)
    conv4 = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv4_2")(conv4)
    drop4 = keras.layers.Dropout(0.2)(conv4) # 64×64×512
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    #=== 32×32×512 ===
    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv5_1")(pool4)
    conv5 = keras.layers.Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv5_2")(conv5)
    drop5 = keras.layers.Dropout(0.2)(conv5)
    up5   = keras.layers.Conv2DTranspose(512, (2,2), strides=2, padding='valid', kernel_initializer = 'he_normal')(drop5)

    """ === expansive path (consists of an upsampling of the feature map.) === """
    #=== 56×56×1024 ===
    crop4  = keras.layers.core.Lambda(lambda x:x[:,4:-4,4:-4,:])(drop4)
    merge6 = keras.layers.Concatenate()([crop4,up5])
    # 56×56×1024
    conv6  = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv6_1")(merge6)
    conv6  = keras.layers.Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv6_2")(conv6)
    up6    = keras.layers.Conv2DTranspose(256, (2,2), strides=2, padding='valid', kernel_initializer = 'he_normal')(conv6)

    #=== 104×104×512 ===
    crop3  = keras.layers.core.Lambda(lambda x:x[:,16:-16,16:-16,:])(conv3)
    merge7 = keras.layers.Concatenate()([crop3,up6])
    # 104×104×512
    conv7  = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv7_1")(merge7)
    conv7  = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv7_2")(conv7)
    up7    = keras.layers.Conv2DTranspose(128, (2,2), strides=2, padding='valid', kernel_initializer = 'he_normal')(conv7)

    #=== 200×200×256 ===
    crop2  = keras.layers.core.Lambda(lambda x:x[:,40:-40,40:-40,:])(conv2)
    merge8 = keras.layers.Concatenate()([crop2,up7])
    # 200×200×256
    conv8  = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv8_1")(merge8)
    conv8  = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv8_2")(conv8)
    up8    = keras.layers.Conv2DTranspose(64, (2,2), strides=2, padding='valid', kernel_initializer = 'he_normal')(conv8)

    #=== 392×392×128 ===
    crop1  = keras.layers.core.Lambda(lambda x:x[:,88:-88,88:-88:])(conv1)
    merge9 = keras.layers.Concatenate()([crop1,up8])
    # 392×392×128
    conv9  = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv9_1")(merge9)
    conv9  = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv9_2")(conv9)
    conv9  = keras.layers.Conv2D(2,  3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal', name="conv9_3")(conv9)

    """ === Output === """
    outputs = keras.layers.Conv2D(1, 1, activation = 'sigmoid', name="output")(conv9)

    model = keras.Model(input = inputs, output = outputs)

    # There is no comment about learning rate and decay in the paper.
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)

    return model
