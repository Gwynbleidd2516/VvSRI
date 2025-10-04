from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def gen_point_clouds(n=100, N=3):
    """Геренирует облака точек"""
    features, labels = make_blobs(n_samples=n*3, centers=N) #Генерирует 3 облака
    scaler = MinMaxScaler((0.1, 1000))
    features = scaler.fit_transform(features) #Масштабирует значения
    return features
