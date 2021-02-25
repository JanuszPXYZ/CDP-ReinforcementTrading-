import numpy as np
from numpy import ndarray
from Agent import *
from helper import *



def test(data: ndarray,
         window_size: int,
         model_name: str):

    done = False
    total_profit = 0
    is_eval = True
    agent = Agent(window_size, is_eval = is_eval, model_name = model_name)
    state = getState(data, 0, window_size + 1)


    l_test = len(data) - 1
    states_sell = []
    states_buy = []

    for t in range(l_test):
        action = agent.act(state)

        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(data[t])
            states_buy.append(t)
            print(f"Buy: {formatPrice(data[t])}")

        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            states_sell.append(t)
            print(f"Sell: {formatPrice(data[t])} | Profit: {formatPrice(data[t] - bought_price)}")

        if t == l_test - 1:
            done = True

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print(f"Total Profit: {formatPrice(total_profit)}")
            plot_decisions(data, states_buy, states_sell, total_profit)
