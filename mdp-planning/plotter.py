import matplotlib.pyplot as plt

# p=[0,0.1,0.2,0.3,0.4,0.5]
# values=[0.7,0.28682,0.19,0.136,0.108,0.1]
# plt.plot(p,values)
# # for i, j in zip(p, values):
# #  plt.annotate('(%s, %s)' % (i, j), xy=(i, j), textcoords='offset points', xytext=(0,10), ha='center')
# plt.grid()
# plt.xlabel("p")
# plt.ylabel("Win probability")
# plt.title("Win probability vs 'p'")
# plt.savefig("plot1.png")

q=[0.6,0.7,0.8,0.9,1]
values=[0.09,0.136,0.21,0.32,0.41]



plt.plot(q,values)
# for i, j in zip(q, values):
#  plt.annotate('(%s, %s)' % (i, j), xy=(i, j), textcoords='offset points', xytext=(0,10), ha='center')
plt.xlabel("q")
plt.grid()
plt.ylabel("Win Probability")
plt.title("Win Probability vs 'q'")
plt.savefig("plot2.png")