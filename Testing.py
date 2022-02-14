from DataHandler import DataHandler
from MLP import task, mlp
import random

def main():
    print("Hello world!")

def train():
    dh = DataHandler("./ETH_USD_1h.csv")
    inputs, outputs, num_stays = dh.generate_batch(n=34000, buy_delta=0.005, sell_delta=0.005)
    print(num_stays)
    #print(inputs)
    traintask = task(inputs, outputs)
    new_mlp = mlp(traintask, 48, load_weights=False)
    new_mlp.train(n_epochs=5000, loss_theta=10, rate_eta=0.5)
    '''for i in range(100):
        rand = round(random.random()*1023)
        new_mlp.feed_forward(inputs[rand])
        print(f"Actual output: {outputs[rand]}")
        print(rand)
    '''
    for i in range(20):
        rand = round(random.random() * 33999)
        print(new_mlp.feed_forward(inputs[rand]))
        print(f"Actual output: {outputs[rand]}")
        print(rand)
    rand = round(random.random() * 1023)
    new_mlp.feed_forward(inputs[92])
    print(f"Actual output: {outputs[92]}")
    print(92)

if __name__ == "__main__":
    train()
