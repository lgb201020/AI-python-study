import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV




#@데이터 불러오기
emails, labels = [], []
paths = ['enron1/spam/','enron1/ham/']
for path in paths:
    for filename in glob.glob(os.path.join(path, '*.txt')):
        with open(filename, 'r', encoding = "ISO-8859-1") as file:
            emails.append(file.read())
            if path.endswith("spam/"):
                labels.append(1)
            else:
                labels.append(0)
#파일을 불러와 데이터를 emails 리스트로 저장하고 labels 리스트에 emails와 동일인덱스 원소에 스팸과 일반메일을 구분해 놓는 1,0을 입력
print(np.unique(labels, return_counts=True))


#@숫자, 구두점, 사람 이름등 스팸메일 분류에 연관이 없는 데이터 분리
nltk.download('names')
nltk.download('wordnet')

all_names = set(nltk.corpus.names.words())
lemmatizer = nltk.stem.WordNetLemmatizer()

cleaned_emails = []

for email in emails:
    cleaned_emails.append(' '.join([lemmatizer.lemmatize(word.lower()) 
                                    for word in email.split()
                                    if word.isalpha() and word not in all_names]))
print(cleaned_emails[0])
"""
* lemmatizer 객체를 만들어 WordNetLemmatizer() 함수를 이용해 사용하지 않은 데이터 불용어 처리
* 모든 단어들을 소문자로 변환하고 단어별로 분리
* All_names에 속하지 않은 단어를 cleaned_emails에 추가한다.
"""


#@불용어 제거와 단어의 출현 빈도 특징을 추출
CountV = CountVectorizer(stop_words="english",max_features=500)
#/*Countvectorizer: 단어 출현 빈도와 관련된 작업을 수행하는 변환기 -> 희소 행렬로 변환되어 출력됨

term_docs = CountV.fit_transform(cleaned_emails)
print(term_docs.shape, term_docs[0])

feature_names = CountV.get_feature_names_out()
print(feature_names[481], feature_names[357], feature_names[125], feature_names[:5])
#*vectorizer.get_feature_names() 메소드를 이용해서 희소행렬로 출력된 CountV를 일반 행렬로 다시 변환한다.


#@훈련 데이터와 테스트 데이터의 분리 및 변환
X_train, X_test, y_train, y_test = train_test_split(cleaned_emails, labels,
                                                    test_size=0.3, random_state=35)
term_docs_train = CountV.fit_transform(X_train)
term_docs_test = CountV.transform(X_test)
#*train data는 모델 fit을 하고 변환하고 (fit_transform(X_train)) test data는 fit을 하지 않고 변환함(transform(X_test))


#@모델의 성능 측정: 정확도, AUC(곡선하면적)
naive_bayes = MultinomialNB(alpha=1, fit_prior=True)
naive_bayes.fit(term_docs_train, y_train)

y_pred = naive_bayes.predict(term_docs_test)
print(y_pred[:5])

print("naive_bayes_score:{}".format(naive_bayes.score(term_docs_test, y_test)))

y_pred_proda = naive_bayes.predict_proba(term_docs_test)
print("입력받은 특징에 대해 y class일 확률".format(y_pred_proda[:5]))


#@모델 성능 시각화
fpr, tpr, unused_element = roc_curve(y_test, y_pred_proda[:,1])
auc = roc_auc_score(y_test, y_pred_proda[:,1])
plt.plot(fpr, tpr, "r-", label="MultinomialNB")
plt.plot([0, 1], [0, 1], "b--", label = "random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curver: AUC={:.5f}".format(auc))
plt.legend(loc = "lower right")
plt.show()


#@모델 성능 개선을 위한 GridSearchCV
params = {
    "alpha" : [i*0.000001 for i in (range(1,2000001))],
    "fit_prior" : [True, False]
}
grid_search = GridSearchCV(naive_bayes, params, n_jobs=1, cv=10, scoring= "roc_auc")
#** pc 설정 및 성능상 n_jobs = 1 즉 single core만 사용하여 gridsearch가 가능하다. 1 이외의 값은 ascii 오류가 생긴다.
grid_search.fit(term_docs_train, y_train)
#데이터 적합

print(grid_search.best_params_)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(term_docs_test)
print(y_pred[:5])

best_model_score = best_model.score(term_docs_test, y_pred)
print(best_model_score)

y_pred_proda = best_model.predict_proba(term_docs_test)
print(y_pred_proda[:5])
#최적 hyperparameter출력 및 최적 모델 객체 선언, 예측 결과, 모델 점수 출력, 모델의 스팸, 일반 메일 판단 확률 출력


#@성능 측정 시각화
fpr, tpr, unused_element = roc_curve(y_test, y_pred_proda[:,1])
auc = roc_auc_score(y_test, y_pred_proda[:,1])
plt.plot(fpr, tpr, "r-", label="MultinomialNB")
plt.plot([0, 1], [0, 1], "b--", label = "random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curver: AUC={:.5f}".format(auc))
plt.legend(loc = "lower right")
plt.show()