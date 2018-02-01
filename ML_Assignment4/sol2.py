# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:13:16 2017

@author: mj
"""

from lea import Lea, B, X, Pf

alternator_b = Lea.boolProb(1 ,1000)
fan_belt_b = Lea.boolProb(2 ,100)


battery_charging = X(alternator_b, fan_belt_b).switch({ (True, True): False,
                                      (False, False ): B(995,1000),
                                      (True, False): False,
                                      (False, True ): False})


battery_flat = Lea.if_(battery_charging, B(10,100), B(90,100))

car_not_start = Lea.if_(battery_flat, True, B(5,100))


print ("Probabilty of broken alternator given car won't start is {0}".format(
        Pf(alternator_b.given(car_not_start))))

print ("Probabilty of battery flat given car won't start is {0}"
       .format(Pf(fan_belt_b.given(car_not_start))))

print ("Probabilty of broken fan belt given car wont start and broken alternator is {0}"
       .format(Pf(fan_belt_b.given(car_not_start & alternator_b))))

print ("Probabilty that broken alternator and fan belt given that car won’t start is {0}"
       .format(Pf((fan_belt_b & alternator_b).given(car_not_start))))