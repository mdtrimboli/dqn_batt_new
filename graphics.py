import matplotlib.pyplot as plt
from numpy import ones
import json


def plotting_reward():
    with open("scores_avg.json", "r") as archivo_json:
        # Carga los datos JSON desde el archivo
        datos = json.load(archivo_json)
    # Extrae las listas de las claves del diccionario
    lista1 = datos.get("dqn_1", [])
    lista2 = datos.get("dqn_2", [])

    # Crea subplots
    plt.figure(figsize=(10, 4))  # Tamaño de la figura

    # Subplot 1
    plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primer subplot
    plt.plot(lista1)
    plt.title("Objective 1")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")

    # Subplot 2
    plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, segundo subplot
    plt.plot(lista2)
    plt.title("Objective 2")
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")

    # Ajusta el espacio entre los subplots
    plt.tight_layout()

    # Muestra el gráfico
    plt.show()

    """
    steps = range(len(rew1))
    plt.plot(steps, rew1, marker='o')
    plt.plot(steps, rew2, marker='o')
    plt.title('Average Rewards for each objective')
    plt.xlabel('Episode number')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend(["Objective 1", "Objective 2"], loc="right")
    plt.show()
    """
    #plt.savefig('curves/reward.png', dpi=600)


def plotting_length(length):
    steps = range(len(length))
    plt.plot(steps, length, marker='o')
    plt.title('Episode Length during training')
    plt.xlabel('Episodes')
    plt.ylabel('Time Steps')
    plt.grid(True)
    plt.legend(["Objective 1", "Objective 2"], loc="right")
    plt.show()
    plt.savefig('curves/length.png', dpi=600)

def plotting_loss(loss):
    steps = range(len(loss))
    plt.plot(steps, loss, marker='o')
    plt.title('Loss')
    plt.xlabel('Loss Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    plt.savefig('curves/loss.png', dpi=600)


def electric_plot(t_curve, v_curve, i_curve, soc_curve):

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(hspace=0.4)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(t_curve, color='tab:orange')
    ax1.plot(45 * ones(len(t_curve)), '--', label='Temperature Limit', color='red')
    ax1.legend(loc="lower right", fontsize='x-small')

    ax2.plot(v_curve, color='tab:orange')
    ax3.plot(soc_curve, color='tab:orange')
    ax3.plot(ones(len(soc_curve)), '--', label='Full Charge Level', color='green')
    ax4.plot(i_curve, color='tab:orange')

    ax1.set_ylim(15, 55.01)
    ax1.set_xlim(0, 252)
    ax2.set_ylim(3.4, 4.6)
    ax2.set_xlim(0, 252)
    ax3.set_ylim(0.2, 1.03)
    ax3.set_xlim(0, 252)
    ax4.set_ylim(-50, 2)
    ax4.set_xlim(0, 252)

    ax1.set_ylabel('Temperature [ºC]')
    ax2.set_ylabel('Voltage [Volts]')
    ax3.set_ylabel('SOC')
    ax4.set_ylabel('Current [Amp.]')

    ax1.set_xlabel('Steps')
    ax2.set_xlabel('Steps')
    ax3.set_xlabel('Steps')
    ax4.set_xlabel('Steps')

    ax1.set_title('Mean Temperature')
    ax2.set_title('Terminal Voltage')
    ax3.set_title('State of Charge')
    ax4.set_title('Action Current')
    ##############################################
    # IMPORTANTE!!!
    #plt.savefig('curves/Eval.png', dpi=600)
    plt.show()


plotting_reward()

