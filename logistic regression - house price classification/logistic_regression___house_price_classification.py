
import numpy as np
import pandas as pd

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.05, epochs=10000, reg_lambda=0.001, normalize=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.normalize = normalize
        self.weights = None
        self.means = None
        self.stds = None
        self.loss_history = []  # պահելու ենք սխալների պատմությունը

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    # ===================== FIT =======================
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        # ⚙️ Նորմալիզացիա
        if self.normalize:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1
            X = (X - self.means) / self.stds

        # bias column
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # Սկզբնական կշիռներ
        self.weights = np.zeros((X.shape[1], 1))

        # 🔁 Ուսուցում
        for i in range(self.epochs):
            z = X @ self.weights
            y_pred = self.sigmoid(z)

            grad = (X.T @ (y_pred - y)) / len(y)
            self.weights -= self.learning_rate * grad

            loss = -np.mean(
                y * np.log(y_pred + 1e-12) +
                (1 - y) * np.log(1 - y_pred + 1e-12)
            )
            self.loss_history.append(loss)

            if i % 2000 == 0:
                print(f"Էպոխ {i}: Կորուստ = {loss:.4f}")

        print("\nՈւսուցումն ավարտվեց ✅")
 
    # ===================== PREDICT =======================
    def predict(self, X, threshold = 0.5):
        X = np.array(X, dtype=float)
        if self.normalize and self.means is not None:
            X = (X - self.means) / self.stds
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # 🧠 Գծային կոմբինացիա և sigmoid
        probs = self.sigmoid(X @ self.weights)

        # 🏷️ Վերադարձնում ենք դաս՝ 0 կամ 1
        return (probs >= threshold).astype(int)


    # ===================== EVALUATE =======================
    def evaluate(self, X, y, threshold=0.5):
        # 1) Համոզվում ենք, որ y-ը Nx1 վեկտոր է և թվային
        y = np.asarray(y, float).reshape(-1, 1)

        # 2) Ստանում ենք մոդելի կանխատեսումները (0/1) նշված շեմով
        y_pred = self.predict(X, threshold=threshold)

        # 3) Հաշվում ենք ճշգրտությունը՝ ճիշտ/ընդամենը
        accuracy = float(np.mean(y_pred == y))

        print(f"Ճշգրտությունը (Accuracy): {accuracy * 100:.2f}%  |  Threshold={threshold}")
        return accuracy
 

# Կարդում ենք houses.csv ֆայլը
df = pd.read_csv("houses.csv")

# Շենքի վիճակ (կարգային արժեքներ)
condition_map = {
    "Վատ": 0,
    "Վերանորոգված": 1,
    "Նորակառույց": 2
}
df["Condition"] = df["Condition"].map(condition_map)

# Թաղամասեր (կատեգորիալ արժեքներ՝ առանց կարգի)
district_map = {
    "Կենտրոն": 4,
    "Արաբկիր": 3,
    "Աջափնյակ": 2,
    "Կոմիտաս": 1,
    "Շենգավիթ": 0,
    "Նոր Նորք": 5
}
df["District"] = df["District"].map(district_map)

# Գինը դարձնում ենք բինար՝ ըստ միջին շեմի
threshold = df["Price"].median()
df["Price"] = (df["Price"] >= threshold).astype(int)

# # Ստուգենք արդյունքը
print(df.head(10))

# Train/Test բաժանում
train = df.iloc[:20]
test = df.iloc[20:]

# # Train և Test հավաքածուների չափերը
# print("Train հավաքածուի չափը:", train.shape)
# print("Test հավաքածուի չափը:", test.shape)

# Փոխակերպում X և y մասերի
X_train = train[["Size", "District", "Condition", "Rooms"]].values
y_train = train["Price"].values.reshape(-1, 1)

X_test = test[["Size", "District", "Condition", "Rooms"]].values
y_test = test["Price"].values.reshape(-1, 1)


# Կառուցում ենք և ուսուցանում
model = LogisticRegressionModel(learning_rate=0.05, epochs=10000)
model.fit(X_train, y_train)

# Գնահատում ենք test տվյալների վրա
y_pred = model.evaluate(X_test, y_test)

# Կանխատեսում նոր տան արժեքի համար
new_house = [[90, 3, 2, 3]]  # [Size, District, Condition, Rooms]
new_house1 = [[65, 1, 0, 2]]

pred = model.predict(new_house1)

if pred == 1:
    print("🏠 Տունը թանկ է 💰")
else:
    print("🏡 Տունը մատչելի է 🏷️")
