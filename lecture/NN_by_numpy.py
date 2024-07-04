import os
import struct
import numpy as np

# 데이터 로드 함수 정의
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>||', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>||||', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshapelen(labels,784)
        images = ((images/255.)- .5)*2
    return images, labels

# 데이터 로드 함수로 데이터 준비
X_train, y_train = load_mnist('')
X_test, y_test = load_mnist('',kind='t10k')

#* NN 알고리즘 정의

class NNMLP(object):
    def __init__(self, n_hidden=30, l2=0, epochs=100, 
                eta=.001,shuffle=True, minibatch_size=1, seed=None):
        ''' 
        @param  minibatch_size=1로 설정하면 각 반복(iteration)마다 단 하나의 데이터 샘플로 신경망을 학습
        이 방법은 **확률적 경사 하강법(Stochastic Gradient Descent, SGD)**이라고도 한다.
        '''
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epoch = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _one_hot(self,y,nclasses):
        onehot = np.zeros((nclasses,y.shape[0]))
        for index, value in enumerate(y.astype(int)):
            #enumerate 함수는 index와 value를 출력하는 함수
            onehot[index, value] = 1

    def _forward(self,X):
        
        #* hidden layer 입력 부분
        z_h = np.dot(X,self.w_h) + self.b_h

        #* hidden layer activation function
        a_h = self._sigmoid(z_h)

        #* output layer 입력 부분
        z_out = np.dot(a_h,self.w_out) + self.b_out

        #* output layer activation function
        a_out = self._softmax(z_out)

        return z_h, a_h, z_out, a_out

    def _sigmoid(self,input):
        output = 1/(1+np.exp(input*-1))
        return output
    
    def _softmax(self,input):
        output = np.exp(input)/sum(np.exp(input))
        return output
    
    def _costfuction(self, y_enc, output):
        L2_terms = (self.l2*(np.sum(self.w_h**2)+np.sum(self.w_out**2)))
        term1 = -y_enc*(np.log(output+1e-10))
        term2 = (1. -y_enc)*(1-np.log(1.- output + 1e-10))

        cost = (1/len(y_enc))*np.sum(term1-term2) +L2_terms

        
        return cost
    def predict(self,X):
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
    