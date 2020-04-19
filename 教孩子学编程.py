import random

secret = 1,2,1,3,8



guess = 0
tries = 0

print("小谢同志！ 我是你的卡哇伊机器人，我有一个秘密！")
print("这个秘密是一个5个数字串的数字，我将给你6次机会。")

while guess != secret and tries < 6 :
    guess =  tuple( map(int,input("你猜这是什么数字呢？").split(',')))
    if guess != secret:
        print("猜错了，你这个小笨蛋！")
        tries = tries +1
        print("你还剩下%d次机会"%(6-tries))
  
    else:    
        print("太棒了！你猜对了，你找到了我的秘密数字串！")
       
    
print("没的再猜了！祝你下次好运")
print("我的秘密数字串是",secret)

