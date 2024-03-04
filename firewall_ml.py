from scapy.all import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset (from pcap file)

def load_dataset(file_path):
    print("load dataset")
    packets = rdpcap(file_path)
    return packets


file_path = 'C:/Users/juika/OneDrive/Desktop/pu hackathon/network_traffic.pcap'
dataset = load_dataset(file_path)


# Step 2: Data Preprocessing
def preprocess_data(packets):
    print("1")
    # Define function to extract features and labels from packets
    # For demonstration purposes, let's assume we're extracting features like packet length, number of packets per second, etc.
    features = []
    labels = []
    for packet in packets:
        # Extract features from each packet
        feature_vector = [len(packet), packet.time]
        features.append(feature_vector)
        
        # Label each packet (you should replace this with actual label extraction logic)
        label = 1 if packet.haslayer(IP) else 0  # Assuming IP packets are legitimate
        labels.append(label)
        
    return features, labels

# Step 3: Split Data into Training and Testing Sets
def split_data(features, labels):
    print("2")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Train Machine Learning Model
def train_model(X_train, y_train):
    print("3")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    print("4")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Step 6: Deploy the Firewall
def deploy_firewall(model, packet):
    print("5")
    # Extract features from the packet
    feature_vector = [len(packet), packet.time]
    
    # Predict the label using the trained model
    predicted_label = model.predict([feature_vector])[0]
    
    # Print the prediction result (for demonstration purposes)
    if predicted_label == 1:
        print("Incoming packet is legitimate.")
    else:
        print("Incoming packet is malicious.")

# Step 7: Main Function
def main():
    # Step 1: Load Dataset
    file_path = 'network_traffic.pcap'
    dataset = load_dataset(file_path)

    # Step 2: Data Preprocessing
    features, labels = preprocess_data(dataset)

    # Step 3: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Step 4: Train Machine Learning Model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate Model Performance
    evaluate_model(model, X_test, y_test)

    # Step 6: Deploy the Firewall (Demo)
    # For demonstration purposes, let's assume we're testing the firewall with the first packet in the dataset
    test_packet = dataset[0]
    deploy_firewall(model, test_packet)

if __name__ == "__main__":
    main()
