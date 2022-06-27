from curses import termattrs
import tensorflow as tf
import tensorflow.keras as keras
import global_vars as gv

class SaveModelCallback(keras.callbacks.Callback):
    
    def __init__(self, freq, model, path=gv.model_path,monitor="val_loss",term=None,term_value=None):
        super().__init__()
        self.freq = freq
        self.model = model
        self.path = path
        self.epoch = 0
        self.min_loss = 10000000
        self.monitor = monitor
        self.term  = term
        self.term_value = term_value
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch+=1
        save = False
        print("current {} is :{}, min {} is:{}".format(self.monitor,logs[self.monitor],self.monitor,self.min_loss))
        if (self.epoch % self.freq == 0 and logs[self.monitor]<self.min_loss):
            if (self.term is not None and self.term_value is not None):
                if (logs[self.term]> self.term_value):
                    save = True
            else:
                save = True
            if save:
                print("saving model {}...".format(self.path))
                self.min_loss =  logs[self.monitor]
                self.model.save(self.path,save_format="tf")
                print("model saved.")
    

# class CustomCallback(keras.callbacks.Callback):
#     def on_train_begin(self, logs=None):
#         keys = list(logs.keys())
#         print("Starting training; got log keys: {}".format(keys))

#     def on_train_end(self, logs=None):
#         keys = list(logs.keys())
#         print("Stop training; got log keys: {}".format(keys))

#     def on_epoch_begin(self, epoch, logs=None):
#         keys = list(logs.keys())
#         print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

#     def on_epoch_end(self, epoch, logs=None):
#         keys = list(logs.keys())
#         print("End epoch {} of training; got log keys: {}".format(epoch, keys))

#     def on_test_begin(self, logs=None):
#         keys = list(logs.keys())
#         print("Start testing; got log keys: {}".format(keys))

#     def on_test_end(self, logs=None):
#         keys = list(logs.keys())
#         print("Stop testing; got log keys: {}".format(keys))

#     def on_predict_begin(self, logs=None):
#         keys = list(logs.keys())
#         print("Start predicting; got log keys: {}".format(keys))

#     def on_predict_end(self, logs=None):
#         keys = list(logs.keys())
#         print("Stop predicting; got log keys: {}".format(keys))

#     def on_train_batch_begin(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

#     def on_train_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

#     def on_test_batch_begin(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

#     def on_test_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

#     def on_predict_batch_begin(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

#     def on_predict_batch_end(self, batch, logs=None):
#         keys = list(logs.keys())
#         print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))