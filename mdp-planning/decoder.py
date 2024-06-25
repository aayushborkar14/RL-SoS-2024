import argparse

parser = argparse.ArgumentParser()

def main():
    parser.add_argument("--value-policy",dest="value",type=str)
    parser.add_argument("--opponent", type=str)


    args = parser.parse_args()
    states=[]
    with open(args.opponent) as file:
        for line in file:
            line2 = line.rstrip()
            words = line2.split()  
            if words[0]=="state":
                continue
            states.append(words[0])
    cou=0
    with open(args.value) as file:
        for line in file:
            line2 = line.rstrip()
            words = line2.split() 
            cou+=1
            if cou<3:
                continue 
            
            print(states[cou-3],end=" ")
            print(words[1],end=" ")
            print(words[0])    
    
if __name__ == "__main__":
    main()
