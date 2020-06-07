# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:15:06 2020

@author: Jay
"""

from __future__ import print_function
import random
import sys


def weighted_random_selection(obj1, obj2):
    """Randomly select between two objects based on assigned 'weight'

    .. todo:: How about creating a utility module for common functionality?
    """
    weighted_list = 6 * [id(obj1)] + 4 * [id(obj2)]
    selection = random.choice(weighted_list)

    if selection == id(obj1):
        return obj1

    return obj2


def print_bold(msg, end='\n'):
    print("\033[1m" + msg + "\033[0m", end=end)


class GameUnit:
    """A base class for creating various game characters"""
    def __init__(self, name=''):
        self.max_hp = 0
        self.health_meter = 0
        self.name = name
        self.enemy = None
        self.unit_type = None

    def info(self):
        """Information on the unit (overridden in subclasses)"""
        pass

    def attack(self, enemy):
        """The main logic to determine injured unit and amount of injury

        .. todo:: Check if enemy exists!
        """
        injured_unit = weighted_random_selection(self, enemy)
        injury = random.randint(10, 15)
        injured_unit.health_meter = max(injured_unit.health_meter - injury, 0)
        print("进攻! ", end='')
        self.show_health(end='  ')
        enemy.show_health(end='  ')

    def heal(self, heal_by=2, full_healing=True):
        """Heal the unit replenishing all the hit points"""
        if self.health_meter == self.max_hp:
            return

        if full_healing:
            self.health_meter = self.max_hp
        else:
            # TODO: Do you see a bug here? it can exceed max hit points!
            self.health_meter += heal_by

        print_bold("你被治疗了!", end=' ')
        self.show_health(bold=True)

    def reset_health_meter(self):
        """Reset the `health_meter` (assign default hit points)"""
        self.health_meter = self.max_hp

    def show_health(self, bold=False, end='\n'):
        """Show the remaining hit points of the player and the enemy"""
        # TODO: what if there is no enemy?
        msg = "生命值: %s: %d" % (self.name, self.health_meter)

        if bold:
            print_bold(msg, end=end)
        else:
            print(msg, end=end)


class Knight(GameUnit):
    """ Class that represents the game character 'Knight'

    The player instance in the game is a Knight instance. Other Knight
    instances are considered as '骑士团队友s' of the player and is
    indicated by the attribute `self.unit_type` .
    """
    def __init__(self, name='谢周峒骑士'):
        super().__init__(name=name)
        self.max_hp = 40
        self.health_meter = self.max_hp
        self.unit_type = '骑士团队友'

    def info(self):
        """Print basic information about this character"""
        print("我是一个骑士!")

    def acquire_hut(self, hut):
        """Fight the combat (command line) to acquire the hut

        .. todo::   acquire_hut method can be refactored.
                   Example: Can you use self.敌人 instead of calling
                   hut.occupant every time?
        """
        print_bold("进入木屋 %d..." % hut.number, end=' ')
        is_enemy = (isinstance(hut.occupant, GameUnit) and
                    hut.occupant.unit_type == '敌人')
        continue_attack = 'y'
        if is_enemy:
            print_bold("敌人 出现!")
            self.show_health(bold=True, end=' ')
            hut.occupant.show_health(bold=True, end=' ')
            while continue_attack:
                continue_attack = input(".......继续攻击? (y/n): ")
                if continue_attack == 'n':
                    self.run_away()
                    break

                self.attack(hut.occupant)

                if hut.occupant.health_meter <= 0:
                    print("")
                    hut.acquire(self)
                    break
                if self.health_meter <= 0:
                    print("")
                    break
        else:
            if hut.get_occupant_type() == '未有人占领':
                print_bold("木屋没有被占领")
            else:
                print_bold("骑士团队友 出现!")
            hut.acquire(self)
            self.heal()

    def run_away(self):
        """Abandon the battle.

        .. seealso:: `self.acquire_hut`
        """
        print_bold("逃跑...")
        self.enemy = None


class OrcRider(GameUnit):
    """Class that represents the game character Orc Rider"""
    def __init__(self, name=''):
        super().__init__(name=name)
        self.max_hp = 30
        self.health_meter = self.max_hp
        self.unit_type = '敌人'
        self.hut_number = 0

    def info(self):
        """Print basic information about this character"""
        print("Grrrr..I am an Orc Wolf Rider. Don't mess with me.")


class Hut:
    """Class to create hut object(s) in the game Attack of the Orcs"""
    def __init__(self, number, occupant):
        self.occupant = occupant
        self.number = number
        self.is_acquired = False

    def acquire(self, new_occupant):
        """Update the occupant of this hut"""
        self.occupant = new_occupant
        self.is_acquired = True
        print_bold("干的好! 木屋 %d 已占领" % self.number)

    def get_occupant_type(self):
        """Return a string giving info on the hut occupant"""
        if self.is_acquired:
            occupant_type = '已占领'
        elif self.occupant is None:
            occupant_type = '未有人占领'
        else:
            occupant_type = self.occupant.unit_type

        return occupant_type






huts = []
player = None

#创造一个骑士实例为玩家
player = Knight()

#创造人物，并占创造一个木屋
for i in range(5):
    choice_lst = ['敌人', '骑士团队友', None]
    computer_choice = random.choice(choice_lst)
    if computer_choice == '敌人':
        name = '敌人-' + str(i+1)
        huts.append(Hut(i+1, OrcRider(name)))#如果选择了敌人就创造带敌人1半兽人的非占领木屋1
    elif computer_choice == '骑士团队友':
        name = 'knight-' + str(i+1)
        huts.append(Hut(i+1, Knight(name)))#如果选择了骑士团队友就创造带骑士团队友1骑士的非占领木屋1
    else:
        huts.append(Hut(i+1, computer_choice))#如果选择了空就创造不带占领者的非占领木屋1


acquired_hut_counter = 0#占领木屋的计数器为0

print_bold("任务:")
print("  1. 和 敌人 战斗.")
print("  2. 控制村里所有的木屋")
print("---------------------------------------------------------\n")
player.show_health(bold=True)

while acquired_hut_counter < 5:
    verifying_choice = True
    idx = 0
    print("目前木屋占领情况: %s" % [x.get_occupant_type() for x in huts])
    while verifying_choice:
        user_choice = input("选择一个木屋号码(1-5): ")
        idx = int(user_choice)
        if huts[idx-1].is_acquired:
            print("你已经占领了该木屋，请重试."
                  "<INFO: 在已占领的木屋里面不能得到治疗.>")
        else:
            verifying_choice = False
    player.acquire_hut(huts[idx-1])   
    if player.health_meter <= 0:
        print_bold("你失败了  :(  下次好运")
        break
    
    if huts[idx-1].is_acquired:
        acquired_hut_counter += 1
    
    if acquired_hut_counter == 5:
        print_bold ("恭喜! 你获得胜利!!!")


