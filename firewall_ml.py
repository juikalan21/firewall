from scapy.all import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_dataset(file_path):
    print("load dataset")
    packets = rdpcap(file_path)
    return packets
file_path = 'C:/Users/juika/OneDrive/Desktop/pu hackathon/network_traffic.pcap'
dataset = load_dataset(file_path)

def preprocess_data(packets):
    print("1")
    features = []
    labels = []
    for packet in packets:
        feature_vector = [len(packet), packet.time]
        features.append(feature_vector)
        label = 1 if packet.haslayer(IP) else 0  # Assuming IP packets are legitimate
        labels.append(label)
    return features, labels

def split_data(features, labels):
    print("2")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("3")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("4")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def deploy_firewall(model, packet):
    print("5")
    feature_vector = [len(packet), packet.time]
    predicted_label = model.predict([feature_vector])[0]
    if predicted_label == 1:
        print("Incoming packet is legitimate.")
    else:
        print("Incoming packet is malicious.")

def main():
    file_path = 'network_traffic.pcap'
    dataset = load_dataset(file_path)
    features, labels = preprocess_data(dataset)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    test_packet = dataset[0]
    deploy_firewall(model, test_packet)

if __name__ == "__main__":
    main()
