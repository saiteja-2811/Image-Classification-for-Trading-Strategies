# Import the Libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
# Part 1 - Data Preprocessing
root_dir = "C:/Users/saite/PycharmProjects/py38"
target_dir = root_dir + "/Image-Classification-for-Trading-Strategies/stockpred/models/"

approach = ['BB','RSI']
methods = ['BPS','BHPS','BPHS']

train_data_dir = str()
validation_data_dir = str()
save_path1 = str()
save_path2 = str()

# Function for the whole model
def model_func(train_data_dir,validation_data_dir,save_path1,save_path2):

    # Preprocessing the Training set
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    training_set = train_datagen.flow_from_directory(train_data_dir,
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')

    # Preprocessing the Test set
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(validation_data_dir,
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')

    # Part 2 - Building the CNN

    # Initialising the CNN
    cnn = tf.keras.models.Sequential()

    # Step 1 - Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    # Step 2 - Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    # Step 3 - Flattening
    cnn.add(tf.keras.layers.Flatten())

    # Step 4 - Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Step 5 - Output Layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    # Part 3 - Training the CNN

    # Compiling the CNN
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',
                                                                             tf.keras.metrics.AUC(),
                                                                             tfa.metrics.F1Score(num_classes=2,average="micro",threshold=0.9)])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    # Training the CNN on the Training set and evaluating it on the Test set
    cnn_model = cnn.fit(x = training_set, validation_data = test_set, epochs = 128,callbacks=[early_stop])

    # Save the model weights
    cnn.save(save_path1)
    cnn.save_weights(save_path2)
    results = cnn_model.summary()
    return results

# Loop for all the possible conditions
for i in approach:
    # BB Approach
    if i == "BB":
        # Loop for all the methods
        for j in methods:
            # Buy & Sell
            if j == "BPS":
                train_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/bb/str1/train/'
                validation_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/bb/str1/val/'
                save_path1 = target_dir + str(i+j+".h5")
                save_path2 = target_dir + str(i+j+"weights.h5")
                model_func(train_data_dir, validation_data_dir, save_path1, save_path2)
            # Buy and [Hold + Sell]
            elif j == "BHPS":
                train_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/bb/str2/train/'
                validation_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/bb/str2/val/'
                save_path1 = target_dir + str(i+j+".h5")
                save_path2 = target_dir + str(i+j+"weights.h5")
                model_func(train_data_dir,validation_data_dir,save_path1,save_path2)
            # [Buy + Hold], Sell
            else:
                train_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/bb/str3/train/'
                validation_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/bb/str3/val/'
                save_path1 = target_dir + str(i+j+".h5")
                save_path2 = target_dir + str(i+j+"weights.h5")
                model_func(train_data_dir,validation_data_dir,save_path1,save_path2)
    else:
        # Loop for all the methods
        for j in methods:
            # Buy & Sell
            if j == "BPS":
                train_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/rsi/str1/train/'
                validation_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/rsi/str1/val/'
                save_path1 = target_dir + str(i+j+".h5")
                save_path2 = target_dir + str(i+j+"weights.h5")
                model_func(train_data_dir, validation_data_dir, save_path1, save_path2)
            # Buy and [Hold + Sell]
            elif j == "BHPS":
                train_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/rsi/str2/train/'
                validation_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/rsi/str2/val/'
                save_path1 = target_dir + str(i+j+".h5")
                save_path2 = target_dir + str(i+j+"weights.h5")
                model_func(train_data_dir,validation_data_dir,save_path1,save_path2)
            # [Buy + Hold], Sell
            else:
                train_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/rsi/str3/train/'
                validation_data_dir = root_dir + '/Image-Classification-for-Trading-Strategies/stockpred/rsi/str3/val/'
                save_path1 = target_dir + str(i+j+".h5")
                save_path2 = target_dir + str(i+j+"weights.h5")
                model_func(train_data_dir,validation_data_dir,save_path1,save_path2)
