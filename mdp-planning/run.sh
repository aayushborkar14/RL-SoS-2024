
python3 encoder.py --opponent data/football/test-2.txt --p $1 --q $2 > football_mdp.txt
python3 planner.py --mdp football_mdp.txt > value.txt
python decoder.py --value-policy value.txt --opponent data/football/test-1.txt > policyfile.txt
cat policyfile.txt | sed -n -e ${3:-1000}p
