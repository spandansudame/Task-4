# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:05:54 2023

@author: Spandan
"""

n=int(input("Enter how many Fibonacci numbers you want to see:"))
num1=0
num2=1
next_number=num2 
count=1
 
while count<=n:
    print(next_number, end=" ")
    count+=1
    num1, num2=num2, next_number
    next_number=num1 + num2
    print()
