from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

MODELS = {
    "logreg": LogisticRegression(max_iter=1000, n_jobs=-1), 
    "tree": DecisionTreeClassifier(random_state=2023), 
    "forest": RandomForestClassifier(n_jobs=-1, random_state=2023),
    "svm": SVC(), 
    "knn": KNeighborsClassifier(n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=2023)
}