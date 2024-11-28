import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, cohen_kappa_score, make_scorer
from sklearn import tree
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
import os

class eval():
    '''
    Use case:
    import eval

    e = eval.eval(train_df_processed, '../eval/test6.txt', 42)
    e.save()

    # result be saved to ../eval/test6.txt
    '''
    def __init__(self, df, path='../eval/test', seed=42):
        self.seed = seed
        self.path = path

        try:
            os.makedirs(self.path)
        except FileExistsError:
            # directory already exists
            pass

        # Data ready, train test spliting
        self.X = df.drop(columns=['sii']).to_numpy()
        self.input_feature = df.drop(columns='sii').columns
        self.y = df['sii'].to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed)

        # evaluation metrics
        def cv_kappa(x, y):
            return cohen_kappa_score(x, y, weights='quadratic')

        self.scoring = {
            'acc': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'f1_macro': 'f1_macro',
            'prec_weighted': 'precision_weighted',
            'prec_macro': 'precision_macro',
            'reca_weighted': 'recall_weighted',
            'reca_macro': 'recall_macro',
            'kappa': make_scorer(cv_kappa),
        }
        
        # self.cv_kappa = make_scorer(cv_kappa)

    def save(self):
        target_names = ['None', 'Mild', 'Moderate', 'Severe']

        # sklearn decision tree
        out = 'Decision tree\n'
        clf1 = DecisionTreeClassifier(random_state=self.seed)
        clf1.fit(self.X_train, self.y_train)
        y_pred = clf1.predict(self.X_train)
        out += 'Training Quadratic Weighted Kappa: ' + str(cohen_kappa_score(self.y_train, y_pred, weights='quadratic')) + '\n'
        out += 'Training report:\n' + classification_report(self.y_train, y_pred, target_names=target_names) + '='*50 + '\n'

        y_pred = clf1.predict(self.X_test)
        out += 'Testing Quadratic Weighted Kappa: ' + str(cohen_kappa_score(self.y_test, y_pred, weights='quadratic')) + '\n'
        out += 'Testing report:\n' + classification_report(self.y_test, y_pred, target_names=target_names) + '='*50 + '\n'

        cv_ls = cross_validate(clf1, self.X, self.y, cv=10, scoring=self.scoring)
        # out += 'Cross_val_score on accuracy:\n' + str(cv_ls['test_acc']) + '\n Mean: ' + str(cv_ls['test_acc'].mean()) + '\n'
        # out += 'Cross_val_score on macro average F1:\n' + str(cv_ls['test_f1_macro']) + '\n Mean: ' + str(cv_ls['test_f1_macro'].mean()) + '\n'
        # out += 'Cross_val_score on weighted average F1:\n' + str(cv_ls['test_f1_weighted']) + '\n Mean: ' + str(cv_ls['test_f1_weighted'].mean()) + '\n'
        # out += 'Cross_val_score on macro averaged precision:\n' + str(cv_ls['test_prec_macro']) + '\n Mean: ' + str(cv_ls['test_prec_macro'].mean()) + '\n'
        # out += 'Cross_val_score on weighted averaged precision:\n' + str(cv_ls['test_prec_weighted']) + '\n Mean: ' + str(cv_ls['test_prec_weighted'].mean()) + '\n'
        # out += 'Cross_val_score on macro averaged recall:\n' + str(cv_ls['test_reca_macro']) + '\n Mean: ' + str(cv_ls['test_reca_macro'].mean()) + '\n'
        # out += 'Cross_val_score on weighted averaged recall:\n' + str(cv_ls['test_reca_weighted']) + '\n Mean: ' + str(cv_ls['test_reca_weighted'].mean()) + '\n'
        # out += 'Cross_val_score on Quadratic Weighted Kappa:\n' + str(cv_ls['test_kappa']) + '\n Mean: ' + str(cv_ls['test_kappa'].mean()) + '\n'

        out += 'Cross val mean report: \n'
        out += f"{'accuracy: ': >30}" + str(cv_ls['test_acc'].mean()) + '\n'
        out += f"{'macro average F1: ': >30}" + str(cv_ls['test_f1_macro'].mean()) + '\n'
        out += f"{'weighted average F1: ': >30}" + str(cv_ls['test_f1_weighted'].mean()) + '\n'
        out += f"{'macro averaged precision: ': >30}" + str(cv_ls['test_prec_macro'].mean()) + '\n'
        out += f"{'weighted averaged precision: ': >30}" + str(cv_ls['test_prec_weighted'].mean()) + '\n'
        out += f"{'macro averaged recall: ': >30}" + str(cv_ls['test_reca_macro'].mean()) + '\n'
        out += f"{'weighted averaged recall: ': >30}" + str(cv_ls['test_reca_weighted'].mean()) + '\n'
        out += f"{'Quadratic Weighted Kappa: ': >30}" + str(cv_ls['test_kappa'].mean()) + '\n'
        out += '='*50 + '\n' + '='*50 + '\n'

        # skleanr random forest
        out += 'Random forest\n'
        clf2 = RandomForestClassifier(random_state=self.seed)
        clf2.fit(self.X_train, self.y_train)
        y_pred = clf2.predict(self.X_train)
        out += 'Training Quadratic Weighted Kappa: ' + str(cohen_kappa_score(self.y_train, y_pred, weights='quadratic'))
        out += 'Training report:\n' + classification_report(self.y_train, y_pred, target_names=target_names) + '='*50 + '\n'

        y_pred = clf2.predict(self.X_test)
        out += 'Testing Quadratic Weighted Kappa: ' + str(cohen_kappa_score(self.y_test, y_pred, weights='quadratic')) + '\n'
        out += 'Testing report:\n' + classification_report(self.y_test, y_pred, target_names=target_names) + '='*50 + '\n'

        cv_ls = cross_validate(clf2, self.X, self.y, cv=10, scoring=self.scoring)
        out += 'Cross val mean report: \n'
        out += f"{'accuracy: ': >30}" + str(cv_ls['test_acc'].mean()) + '\n'
        out += f"{'macro average F1: ': >30}" + str(cv_ls['test_f1_macro'].mean()) + '\n'
        out += f"{'weighted average F1: ': >30}" + str(cv_ls['test_f1_weighted'].mean()) + '\n'
        out += f"{'macro averaged precision: ': >30}" + str(cv_ls['test_prec_macro'].mean()) + '\n'
        out += f"{'weighted averaged precision: ': >30}" + str(cv_ls['test_prec_weighted'].mean()) + '\n'
        out += f"{'macro averaged recall: ': >30}" + str(cv_ls['test_reca_macro'].mean()) + '\n'
        out += f"{'weighted averaged recall: ': >30}" + str(cv_ls['test_reca_weighted'].mean()) + '\n'
        out += f"{'Quadratic Weighted Kappa: ': >30}" + str(cv_ls['test_kappa'].mean()) + '\n'
        out += '='*50 + '\n' + '='*50 + '\n'

        # extreme gradient boost for random forest
        out += 'Extreme gradient boost for random forest\n'
        clf3 = xgb.XGBRFClassifier(random_state=42)
        clf3.fit(self.X_train, self.y_train)
        y_pred = clf3.predict(self.X_train)
        out += 'Training Quadratic Weighted Kappa: ' + str(cohen_kappa_score(self.y_train, y_pred, weights='quadratic')) + '\n'
        out += 'Training report:\n' + classification_report(self.y_train, y_pred, target_names=target_names) + '='*50 + '\n'

        y_pred = clf3.predict(self.X_test)
        out += 'Testing Quadratic Weighted Kappa: ' + str(cohen_kappa_score(self.y_test, y_pred, weights='quadratic')) + '\n'
        out += 'Testing report:\n' + classification_report(self.y_test, y_pred, target_names=target_names) + '='*50 + '\n'

        cv_ls = cross_validate(clf3, self.X, self.y, cv=10, scoring=self.scoring)        
        out += 'Cross val mean report: \n'
        out += f"{'accuracy: ': >30}" + str(cv_ls['test_acc'].mean()) + '\n'
        out += f"{'macro average F1: ': >30}" + str(cv_ls['test_f1_macro'].mean()) + '\n'
        out += f"{'weighted average F1: ': >30}" + str(cv_ls['test_f1_weighted'].mean()) + '\n'
        out += f"{'macro averaged precision: ': >30}" + str(cv_ls['test_prec_macro'].mean()) + '\n'
        out += f"{'weighted averaged precision: ': >30}" + str(cv_ls['test_prec_weighted'].mean()) + '\n'
        out += f"{'macro averaged recall: ': >30}" + str(cv_ls['test_reca_macro'].mean()) + '\n'
        out += f"{'weighted averaged recall: ': >30}" + str(cv_ls['test_reca_weighted'].mean()) + '\n'
        out += f"{'Quadratic Weighted Kappa: ': >30}" + str(cv_ls['test_kappa'].mean()) + '\n'
        out += '='*50 + '\n' + '='*50 + '\n'

        with open(self.path + '/evaluation_score.txt', 'w') as f:
            f.write(out)

        plt.figure(figsize=(30,12))
        tree.plot_tree(clf1, filled=True, rounded=True, max_depth=3, fontsize=10, feature_names=self.input_feature)
        plt.title("Decision tree trained on Feature")
        plt.savefig(self.path + '/DT.png')

        print('ended, result at: ', self.path)                                                      
