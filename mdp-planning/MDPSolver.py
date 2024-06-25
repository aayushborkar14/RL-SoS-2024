import numpy as np
from pulp import *

class MDPSolver():
    def __init__(self, mdp_file, algo, policy="None"):
        self.adj = []
        self.num_states = 0
        self.num_actions = 0
        self.gamma=0
        self.end_states=[]
        self.mdptype="default"

            
        with open(mdp_file) as file:
            for line in file:
                line2 = line.rstrip()
                words = line2.split()
                if words[0] == "transition":
                     s1 = int(words[1])
                     ac = int(words[2])
                     s2 = int(words[3])
                     r = float(words[4])
                     p = float(words[5])
                     self.adj[s1][ac].append([s2,p,r])                    
                elif words[0] == "numStates":
                    self.num_states = int(words[1])
                elif words[0] == "numActions":
                    self.num_actions = int(words[1])
                    # self.adj = [[[]*self.num_states]*self.num_states]
                    for i in range(self.num_states):
                        v=[]
                        for i in range(self.num_actions):
                            v.append([])
                        self.adj.append(v)
                elif words[0] == "end":
                    if words[1]!="-1":
                        self.end_states = [int(words[i]) for i in range(1,len(words))]
                elif words[0] == "discount":
                    self.gamma = float(words[1])
                else:
                    self.mdptype = words[1]
                    

        if policy!=None:
            policies=[]
            with open(policy) as file:
                for line in file:
                    line2 = line.rstrip()
                    words = line2.split()
                    policies.append(int(words[0]))
            self.policy_eval(policies)
            return 
        
        if algo=="vi":
            self.value_iteration()
        elif algo=="hpi":
            self.policy_iteration()
        else :
            self.linear_programming()
       
       #HOWARD policy iteration 
    def policy_iteration(self):
        policies=np.random.randint(self.num_actions,size=(self.num_states))
        values_old=self.policy_eval_return(policies)
        flag=False
        while flag==False:
            flag=True
            values_new=values_old
            for s in range(self.num_states):
                for ac in range(self.num_actions):
                    new_val=self.evaluate(s,ac,values_old)
                    if new_val>values_old[s]:
                        flag=False
                        values_new[s]=new_val
                        policies[s]=ac
                        # break
            values_old=values_new
        for i in range(self.num_states):
            print(values_old[i],end=" ")
            print(policies[i])
                    
                    
    def linear_programming(self):
        model=pulp.LpProblem("OptimalValueFunction",LpMinimize)
        decision_variables=[]
        mapper={}
        for i in range(self.num_states):
            var=str('V'+str(i))
            var=pulp.LpVariable(var)
            decision_variables.append(var)
            mapper[var]=i
        #objective
        objective=""
        for i in decision_variables:
            objective+=i
        model+=objective
        #constraints
        for s in range(self.num_states):
            for ac in range(self.num_actions):
                lhs=""
                rhs=0
                lhs+=1*decision_variables[s]
                for i in range(len(self.adj[s][ac])):
                    trans=self.adj[s][ac][i]
                    s2=trans[0]
                    p=trans[1]
                    reward=trans[2]
                    lhs+=(-1*p*self.gamma*decision_variables[s2])
                    rhs+=(p*reward)
                # rhs=str(rhs)
                model+= (lhs>=rhs)
        # result=model.solve(msg=0)
        model.solve(PULP_CBC_CMD(msg=0))
        
        old_values=[None]*self.num_states
        for v in model.variables():
            old_values[mapper[v]]=(v.varValue)
        policies=[]
        for s in range(self.num_states):
            policies.append(np.argmax([self.evaluate(s,ac,old_values) for ac in range(self.num_actions)]))
        for i in range(self.num_states):
            print(old_values[i],end=" ")
            print(policies[i])    
            
            

    def value_iteration(self):
        values_old = [0.0]*self.num_states
        theta=1e-10
        policies=[0]*self.num_states
        while True:
            delta = 0.0
            for s in range(self.num_states):
                temp=values_old[s]
                for ac in range(self.num_actions):
                    new_val=self.evaluate(s,ac,values_old)
                    if new_val>values_old[s]:
                        values_old[s]=new_val
                        policies[s]=ac
                delta=max(delta,abs(temp-values_old[s]))
            if delta<theta:
                break
        
        for i in range(self.num_states):
            print(values_old[i],end=" ")
            print(policies[i])
            
                  
       
    
    def policy_eval(self,policies):
        theta=1e-10
        
        
        values_old = [0.0]*self.num_states
        while True:
            delta = 0.0
            for s in range(self.num_states):
                temp=values_old[s]
                values_old[s]=self.evaluate(s,policies[s],values_old)
                delta=max(delta,abs(values_old[s]-temp))
            if delta<theta:
                break
        for i in range(self.num_states):
            print(values_old[i],end=" ")
            print(policies[i])
            
    def policy_eval_return(self,policies):
        theta=1e-10
        
        values_old = [0.0]*self.num_states
        while True:
            delta = 0.0
            for s in range(self.num_states):
                temp=values_old[s]
                values_old[s]=self.evaluate(s,policies[s],values_old)
                delta=max(delta,abs(values_old[s]-temp))
            if delta<theta:
                break
        return values_old
        
        
      # Q-value computation  
    def evaluate(self,s,ac,values_old):
        ans=0
        for i in range(len(self.adj[s][ac])):
            trans=self.adj[s][ac][i]
            s2=trans[0]
            p=trans[1]
            reward=trans[2]
            ans+=(p*reward)
            ans+=(p*self.gamma*values_old[s2])
        return ans
        
        
        
        
                    
                    




