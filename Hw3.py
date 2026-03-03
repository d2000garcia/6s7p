file = open("D:\Diego\Manuals\ANU_num.txt",'r')
numbers = []
for line in file:
    temp = line.strip()
    numbers.append(list(map(int,list(temp))))
sums = list(map(sum,numbers))
means = list(map(lambda x,y: x/y, sums, [10,100,1000]))
for i in range(len(numbers)):
    print("For length %i:"%(len(numbers[i])))
    print("     0s: %i, 1s: %i"%(len(numbers[i])-sums[i],sums[i]))
    print("     Mean: %0.6f"%(means[i]))