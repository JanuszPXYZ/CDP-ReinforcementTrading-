import numpy as np
from Agent import *
from helper import *
from numpy import ndarray


def train(data: ndarray,
          window_size: int,
          agent: Agent,
          batch_size: int,
          epochs: int = 3):

    l = len(data) - 1
    states_buy = []
    states_sell = []


    for e in range(epochs + 1):
        print(f"Episode:{e} / {epochs}")

        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)

            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1: # buy
                agent.inventory.append(data[t])
                states_buy.append(t)
                print(f"Buy: {formatPrice(data[t])}")

            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)

                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price

                states_sell.append(t)
                print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

            done = True if t == l - 1 else False

            next_state = getState(data, t + 1, window_size + 1)

            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("-----------------------------------")
                print(f"Total Profit: {formatPrice(total_profit)}")
                print("-----------------------------------")
                plot_decisions(data, states_buy, states_sell, total_profit)


            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
