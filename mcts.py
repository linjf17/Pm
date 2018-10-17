#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import queue
import random

import numpy as np

from . import BaseAgent

import sys
import math
import random
import queue
import numpy as np
import time
from .. import constants
from .. import utility
from collections import defaultdict

class SimpleAgent(BaseAgent):
    """This is a baseline agent. After you can beat it, submit your agent to
    compete.
    """

    def __init__(self, *args, **kwargs):
        super(SimpleAgent, self).__init__(*args, **kwargs)

        # Keep track of recently visited uninteresting positions so that we
        # don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        # Keep track of the previous direction to help with the enemy standoffs.
        self._prev_direction = None

    def act(self, obs):
        def convert_bombs(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                })
            return ret

        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        bombs = convert_bombs(np.array(obs['bomb_blast_strength']))
        enemies = [constants.Item(e) for e in obs['enemies']]
        ammo = int(obs['ammo'])
        blast_strength = int(obs['blast_strength'])
        items, dist, prev = self._djikstra(
            board, my_position, bombs, enemies, depth=10)

        # Move if we are in an unsafe place.
        unsafe_directions = self._directions_in_range_of_bomb(
            board, my_position, bombs, dist)
        if unsafe_directions:
            directions = self._find_safe_directions(
                board, my_position, unsafe_directions, bombs, enemies)
            return random.choice(directions).value

        # Lay pomme if we are adjacent to an enemy.
        if self._is_adjacent_enemy(items, dist, enemies) and self._maybe_bomb(
                ammo, blast_strength, items, dist, my_position):
            return constants.Action.Bomb.value

        # Move towards an enemy if there is one in exactly three reachable spaces.
        direction = self._near_enemy(my_position, items, dist, prev, enemies, 3)
        if direction is not None and (self._prev_direction != direction or
                                      random.random() < .5):
            self._prev_direction = direction
            return direction.value

        # Move towards a good item if there is one within two reachable spaces.
        direction = self._near_good_powerup(my_position, items, dist, prev, 2)
        if direction is not None:
            return direction.value

        # Maybe lay a bomb if we are within a space of a wooden wall.
        if self._near_wood(my_position, items, dist, prev, 1):
            if self._maybe_bomb(ammo, blast_strength, items, dist, my_position):
                return constants.Action.Bomb.value
            else:
                return constants.Action.Stop.value

        # Move towards a wooden wall if there is one within two reachable spaces and you have a bomb.
        direction = self._near_wood(my_position, items, dist, prev, 2)
        if direction is not None:
            directions = self._filter_unsafe_directions(board, my_position,
                                                        [direction], bombs)
            if directions:
                return directions[0].value

        # Choose a random but valid direction.
        directions = [
            constants.Action.Stop, constants.Action.Left,
            constants.Action.Right, constants.Action.Up, constants.Action.Down
        ]
        valid_directions = self._filter_invalid_directions(
            board, my_position, directions, enemies)
        directions = self._filter_unsafe_directions(board, my_position,
                                                    valid_directions, bombs)
        directions = self._filter_recently_visited(
            directions, my_position, self._recently_visited_positions)
        if len(directions) > 1:
            directions = [k for k in directions if k != constants.Action.Stop]
        if not len(directions):
            directions = [constants.Action.Stop]

        # Add this position to the recently visited uninteresting positions so we don't return immediately.
        self._recently_visited_positions.append(my_position)
        self._recently_visited_positions = self._recently_visited_positions[
            -self._recently_visited_length:]

        return random.choice(directions).value

    @staticmethod
    def _djikstra(board, my_position, bombs, enemies, depth=None, exclude=None):
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1, p_2):
            '''Determines if two points are out of rang of each other'''
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist = {}
        prev = {}
        Q = queue.Queue()

        my_x, my_y = my_position
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth)):
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth)):
                position = (r, c)
                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude),
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)

                if position == my_position:
                    Q.put(position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf


        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            position = Q.get()

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                val = dist[(x, y)] + 1
                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + x, col + y)
                    if new_position not in dist:
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position
                        Q.put(new_position)
                    elif (val == dist[new_position] and random.random() < .5):
                        dist[new_position] = val
                        prev[new_position] = position


        return items, dist, prev

    def _directions_in_range_of_bomb(self, board, my_position, bombs, dist):
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                        constants.Action.Right,
                        constants.Action.Left,
                        constants.Action.Up,
                        constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])
            elif x == position[0]:
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    # Bomb is down.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _find_safe_directions(self, board, my_position, unsafe_directions,
                              bombs, enemies):

        def is_stuck_direction(next_position, bomb_range, next_board, enemies):
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board,
                                                        new_position, enemies):
                        continue

                    dist = abs(row + position_x - next_x) + abs(col + position_y - next_y)
                    Q.put((dist, new_position))
            return is_stuck

        # All directions are unsafe. Return a position that won't leave us locked.
        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(
                    my_position, direction)
                next_x, next_y = next_position
                if not utility.position_on_board(next_board, next_position) or \
                   not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board,
                                          enemies):
                    # We found a direction that works. The .items provided
                    # a small bit of randomness. So let's go with this one.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []  # The directions that will go off the board.

        for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row, y + col)
            direction = utility.get_direction(my_position, position)

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position,
                                            enemies) or utility.position_is_fog(
                                                board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed.
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    @staticmethod
    def _is_adjacent_enemy(items, dist, enemies):
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    @staticmethod
    def _has_bomb(obs):
        return obs['ammo'] >= 1

    @staticmethod
    def _maybe_bomb(ammo, blast_strength, items, dist, my_position):
        """Returns whether we can safely bomb right now.

        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # Will we be stuck?
        x, y = my_position
        for position in items.get(constants.Item.Passage):
            if dist[position] == np.inf:
                continue

            # We can reach a passage that's outside of the bomb strength.
            if dist[position] > blast_strength:
                return True

            # We can reach a passage that's outside of the bomb scope.
            position_x, position_y = position
            if position_x != x and position_y != y:
                return True

        return False

    @staticmethod
    def _nearest_position(dist, objs, items, radius):
        nearest = None
        dist_to = max(dist.values())

        for obj in objs:
            for position in items.get(obj, []):
                d = dist[position]
                if d <= radius and d <= dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    @staticmethod
    def _get_direction_towards_position(my_position, position, prev):
        if not position:
            return None

        next_position = position
        while prev[next_position] != my_position:
            next_position = prev[next_position]

        return utility.get_direction(my_position, next_position)

    @classmethod
    def _near_enemy(cls, my_position, items, dist, prev, enemies, radius):
        nearest_enemy_position = cls._nearest_position(dist, enemies, items,
                                                       radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_enemy_position, prev)

    @classmethod
    def _near_good_powerup(cls, my_position, items, dist, prev, radius):
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange,
            constants.Item.Kick
        ]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @classmethod
    def _near_wood(cls, my_position, items, dist, prev, radius):
        objs = [constants.Item.Wood]
        nearest_item_position = cls._nearest_position(dist, objs, items, radius)
        return cls._get_direction_towards_position(my_position,
                                                   nearest_item_position, prev)

    @staticmethod
    def _filter_invalid_directions(board, my_position, directions, enemies):
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(
                    board, position) and utility.position_is_passable(
                        board, position, enemies):
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_unsafe_directions(board, my_position, directions, bombs):
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bomb_x, bomb_y = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bomb_x and abs(bomb_y - y) <= blast_strength) or \
                   (y == bomb_y and abs(bomb_x - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    @staticmethod
    def _filter_recently_visited(directions, my_position,
                                 recently_visited_positions):
        ret = []
        for direction in directions:
            if not utility.get_next_position(
                    my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret



t_agent = SimpleAgent()
class State(object):
    """
    蒙特卡罗树搜索的游戏状态，记录在某一个Node节点下的状态数据，包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
    需要实现判断当前状态是否达到游戏结束状态，支持从Action集合中随机取出操作。
    """

    def __init__(self, obs, mode):
        self.current_value = 0.0
        # For the first root node, the index is 0 and the game should start from 1
        self.move = 0  # 用来记录这个state自己是怎么动的（譬如向上还是放炸弹
        self.obs = obs
        self.can_kick = obs['can_kick']
        self.my_position = tuple(obs['position'])
        self.board = np.array(obs['board'])
        self.flame_board = obs['flame_board']
        self.bombs = np.array(obs['bomb_blast_strength'])  # bombs是爆炸半径
        self.bombs_life_time = np.array(obs['bomb_life'])
        self.enemies = [constants.Item(e) for e in obs['enemies']]
        self.mode = mode  # 0是逃跑，1是进攻
        self.target_enemy = None
        self.blast_strength = obs['blast_strength']

        # self.actions = None

    def set_target_enemy(self, enemy):
        self.target_enemy = enemy

    def get_available_action(self):
        return self.available_action(self.board)

    def position_is_passable(self, board, flame_board, position):
        x, y = position
        if any([len(board) <= x, len(board[0]) <= y, x < 0, y < 0]):
            return False
        availabe_choice = [0, 5, 6, 7, 8, 9]
        # if self.can_kick:
        #     availabe_choice.append(3)
        if board[x, y] in availabe_choice or flame_board[x][y] == 1:
            return True
        else:
            return False

    def available_action(self, board):

        def is_stuck_direction(next_position, next_board, enemies, blast_strength, my_position):
            bomb_range = blast_strength
            '''Helper function to do determine if the agents next move is possible.'''
            Q = queue.PriorityQueue()
            Q.put((1, next_position))
            seen = set()

            my_x, my_y = my_position

            tmpt_board = next_board.copy()
            tmpt_board[my_x][my_y] = 1

            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                if my_x != position_x and my_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range:
                    is_stuck = False
                    break

                for row, col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row + position_x, col + position_y)
                    if new_position in seen:
                        continue

                    if not self.position_is_passable(tmpt_board,self.flame_board,
                                                     new_position):
                        continue

                    dist = abs(row + position_x - my_x) + abs(col + position_y - my_y)
                    Q.put((dist, new_position))
            return is_stuck

        row, col = self.my_position
        actions = []#因为现在进入evade的条件是周围可能被炸到，因此不动也是选项之一
        if self.mode == 1:
            erow, ecol = self.target_enemy[0]
            canKill = abs(row-erow+col-ecol) < self.blast_strength
            if (row == erow or col == ecol) and canKill:
                actions.append(5)
            elif(abs(erow - row) + abs(ecol - col) < 3):
                actions.append(5)
        i = 0
        for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            i += 1
            new_position = (row + drow, col + dcol)

            next_board = self.board.copy()
            blast_strength = 0

            if self.bombs[row, col] != 0:
                blast_strength = self.bombs[row, col] - 1  # 因为爆炸为2的时候实际上只涉及周围一格
                next_board[self.my_position] = constants.Item.Bomb.value

            if self.position_is_passable(next_board,self.flame_board, new_position):
                if not is_stuck_direction(new_position, next_board, self.enemies, blast_strength, self.my_position):
                    actions.append(i)
        if not actions:
            actions.append(random.randint(0, 4))

        return actions

    def convert_bombs(self, bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret

    def convert_bombs_life_time(self, bomb_map):
        '''Flatten outs the bomb array'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'bombs_life_time': int(bomb_map[(r, c)])
            })
        return ret

    def get_current_obs(self):
        return self.obs

    def get_current_board(self):
        return self.board

    def set_current_board(self, board):
        self.board = board

    def get_current_bombs(self):
        return self.bombs

    def set_current_bombs(self, bombs):
        self.bombs = bombs

    def get_current_bombs_life_time(self):
        return self.bombs_life_time

    def set_current_bombs_life_time(self, bombs_life_time):
        self.bombs_life_time = bombs_life_time

    def get_current_position(self):
        return self.my_position

    def set_current_position(self, position):
        self.position = position

    def get_current_value(self):
        return self.current_value

    def set_current_value(self, value):
        self.current_value = value

    def get_current_move(self):
        return self.move

    def get_current_mode(self):
        return self.mode

    def set_current_move(self, move):
        self.move = move

    def is_terminal(self):
        alive = [False for i in range(4)]
        check = np.where(self.board == 10)
        if check[0].size == 0:
            return 1  # 自己已经被干掉了
        for j in range(3):
            tmptId = 11 + j
            check = np.where(self.get_current_board() == tmptId)
            if check[0].size == 0:
                alive[j + 1] = True
        if alive[1:] == [True, True, True]:
            return 2  # 说明其他人都被干掉了
        return 3  # 游戏继续

    def compute_reward(self):

        if self.is_terminal() == 1:
            return -40
        if self.is_terminal() == 2:
            return 150
        if self.get_current_mode() == 0:

            bomb_map = self.bombs.copy()
            def cross(x, y, radius):
                for xx in range(x-radius, x+radius+1):
                    if 0 <= xx < 11 and bomb_map[xx][y] == 0:
                        bomb_map[xx][y] = 1
                for yy in range(y-radius, y+radius+1):
                    if 0 <= yy < 11 and bomb_map[x][yy] == 0:
                        bomb_map[x][yy] = 1

            for x in range(11):
                for y in range(11):
                    if (bomb_map[x][y] > 1):
                        cross(x, y, int(bomb_map[x][y]-1))
            # print(bomb_map)
            n = 3
            floodfill_area = 1
            safety_area = 0
            my_row, my_col = self.my_position
            for i in range(0, 2 * n):
                for j in range(0, 2 * n):
                    # target表示agent周围的区域
                    target_row, target_col = my_row - n + i, my_col - n + j
                    target_position = (target_row, target_col)
                    if abs(i - n) + abs(j - n) <= n and -1<target_col<11 and -1<target_row<11:
                        
                        # safe
                        if self.position_is_passable(self.board, self.flame_board, target_position):
                            floodfill_area += 1
                            if (bomb_map[target_row][target_col] != 1):
                                safety_area += 1

            free =  safety_area / floodfill_area #让自己走到一个比较宽敞的位置
           # evade
            total = 0
            x, y = self.my_position
            tmptBomb = self.convert_bombs(self.bombs)
            for bomb in tmptBomb:
                position = bomb['position']
                bomb_range = bomb['blast_strength']
                x1, y1 = position
                distance = abs(x1 - x + y1 - y)
                if distance >= bomb_range or self.get_current_bombs_life_time()[x1][y1] > 9:
                    continue

                if x == x1 or y == y1:
                    total += 25 * (10 - self.get_current_bombs_life_time()[x1][y1]) / 9

            # print("original", (100-total), "free", ((100-total)*free/90), "now", (100 - total + (100-total)*free))
            #根据evade的得分的规模来决定加上多少free?
            return ((100 - total)*(1+free/2))
            
        else:
            total = 0
            x, y = self.my_position
            tmptBomb = self.convert_bombs(self.bombs)
            for bomb in tmptBomb:
                position = bomb['position']
                bomb_range = bomb['blast_strength']
                x1, y1 = position
                distance = abs(x1 - x + y1 - y)
                if distance >= bomb_range or self.get_current_bombs_life_time()[x1][y1] > 5:
                    continue

                if x == x1 or y == y1:
                    total += 25 * (6 - self.get_current_bombs_life_time()[x1][y1]) /6#进攻的时候只会管5以下的炸弹
            # attack
            bomb_map = self.bombs.copy()
            def cross(x, y, radius):
                for xx in range(x-radius, x+radius+1):
                    if 0 <= xx < 11 and bomb_map[xx][y] == 0:
                        bomb_map[xx][y] = 1
                for yy in range(y-radius, y+radius+1):
                    if 0 <= yy < 11 and bomb_map[x][yy] == 0:
                        bomb_map[x][yy] = 1

            for x in range(11):
                for y in range(11):
                    if (bomb_map[x][y] > 1):
                        cross(x, y, int(bomb_map[x][y]-1))


            enemy, n, enemy_id = self.target_enemy
            n = 2
            floodfill_area = 1#防止最后除以0
            safety_area = 0
            enemy_row, enemy_col = enemy
            enemy_down = True
            for i in range(11):
                for j in range(11):
                    if self.board[i][j] == enemy_id:
                        enemy_row = i
                        enemy_col = j
                        enemy_down = False

            if enemy_down:
                return 100#干掉了我们的目标敌人

            
            for i in range(0, 2 * n):
                for j in range(0, 2 * n):
                    # target表示agent周围的区域
                    target_row, target_col = enemy_row - n + i, enemy_col - n + j
                    target_position = (target_row, target_col)
                    if abs(i - n) + abs(j - n) <= n and -1<target_col<11 and -1<target_row<11:
                        
                        # safe
                        if self.position_is_passable(self.board, self.flame_board, target_position):
                            floodfill_area += 1
                            if (bomb_map[target_row][target_col] != 1):
                                safety_area += 1
            # print("bomb map", bomb_map)
            # print("enemy pos", enemy)
            # print("Safe", safety_area)
            # print("floodfill_area", floodfill_area)
            return ((100) * (1 - safety_area / floodfill_area))

    def get_next_state_with_random_choice(self):
        # print(self.obs)
        global t_agent
        my_pos = self.obs['position']
        my_id = self.obs['board'][my_pos[0]][my_pos[1]]
        # print(my_id)
        enemy_arr = [e.value for e in self.enemies]
        # print(enemy_arr)
        actions = [0, 0, 0, 0]
        actions[my_id - 10] = random.choice(self.get_available_action())

        for i in range(10, 14):
            if i == my_id:
                continue
            if i not in self.obs['alive']:
                continue
            tmpt_obs = self.obs.copy()
            # print(type(tmpt_obs['position']))
            tmpt_obs['position'] = np.where(tmpt_obs['board'] == i)
            tmpt_obs['position'] = tuple([tmpt_obs['position'][0][0], tmpt_obs['position'][1][0]])
            # print(tmpt_obs['position'])
            tmpt_obs['blast_strength'] = self.blast_strength
            tmpt_obs['enemies'] = []
            for j in range(10, 14):
                if j == i:
                    continue
                tmpt_obs['enemies'].append(constants.Item(j))
            # print(tmpt_obs)
            actions[i - 10] = t_agent.act(tmpt_obs)

        # print(actions)
        # input()
        '''
        actions.append(random.choice(self.get_available_action()))
        for i in range(3):
            actions.append(random.randint(0, 5))
        '''

        tmpt_board = self.board.copy()
        # print(tmpt_board)
        tmpt_bombs = self.bombs.copy()
        tmpt_flame_board = self.flame_board.copy()
        tmpt_bombs_life_time = self.bombs_life_time.copy()


        #判断是否会bounceback
        bounce_back = [False for i in range(4)]
        next_pos = [(-1, -1) for i in range(4)]
        tmpt_id =  10
        for action in actions:
            find_agent = np.where(tmpt_board == tmpt_id)
            if find_agent[0].size > 0:
                agentX = find_agent[0][0]
                agentY = find_agent[1][0]
                next_pos[tmpt_id-10] = (agentX,agentY)
                if action in [0, 1, 2, 3, 4]:
                    x = [0, -1, 1, 0, 0]
                    y = [0, 0, 0, -1, 1]
                    if (-1 < agentX + x[action] < 11) and (-1 < agentY + y[action] < 11):
                        next_pos[tmpt_id-10] = (agentX + x[action],agentY + y[action])
            tmpt_id += 1

        for i in range(3):
            if next_pos[i] == (-1,-1):
                bounce_back[i] = True#如果这个agent死了，就当bounceback
            for j in range(i + 1, 4):
                if next_pos[i] == next_pos[j]:
                    bounce_back[i] = True
                    bounce_back[j] = True

        # 先走再更新炸弹
        id = 10
        next_state = State(self.get_current_obs(), self.get_current_mode())
        for action in actions:
            find_agent = np.where(tmpt_board == id)
            if find_agent[0].size > 0:
                agentX = find_agent[0][0]
                agentY = find_agent[1][0]
                if (id == my_id):
                    next_state.my_position = (agentX, agentY)
                if action == 5:
                    tmpt_bombs[agentX][agentY] = self.blast_strength  # 注意默认的爆炸半径是2
                    tmpt_bombs_life_time[agentX][agentY] = 10
                else:
                    if not bounce_back[id-10]:
                        x,y = next_pos[id-10]
                        if tmpt_board[x][y] == 0:
                            tmpt_board[x][y] = id
                            tmpt_board[agentX][agentY] = 0
            id += 1

        #（连环）引爆函数
        visited = set()
        def cross_fire(x, y, radius):
            visited.add((x,y))
            tmpt_flame_board[x][y] = 4
            for i in range(1, radius+1):
                xx = x - i
                if 0 <= xx < 11:
                    if (tmpt_board[xx][y] in (1,5)):
                        break
                    elif tmpt_board[xx][y] == 3:
                        if  (xx,y) not in visited:
                            cross_fire(xx, y, int(tmpt_bombs[xx][y]-1))
                        break
                    elif tmpt_board[xx][y] in (0,4):
                        tmpt_flame_board[xx][y] = 4
                    else:
                        tmpt_flame_board[xx][y] = 4
                        break

            for i in range(1, radius+1):
                xx = x + i
                if 0 <= xx < 11:
                    if (tmpt_board[xx][y] in (1,5)):
                        break
                    elif tmpt_board[xx][y] == 3:
                        if  (xx,y) not in visited:
                            cross_fire(xx, y, int(tmpt_bombs[xx][y]-1))
                        break
                    elif tmpt_board[xx][y] in (0,4):
                        tmpt_flame_board[xx][y] = 4
                    else:
                        tmpt_flame_board[xx][y] = 4
                        break

            for i in range(1, radius+1):
                yy = y - i
                if 0 <= yy < 11:
                    if (tmpt_board[x][yy] in (1, 5)):
                        break
                    elif  tmpt_board[x][yy] == 3:
                        if (x, yy) not in visited:
                            cross_fire(x, yy, int(tmpt_bombs[x][yy]-1))
                        break
                    elif tmpt_board[x][yy] in (0,4):
                        tmpt_flame_board[x][yy] = 4
                    else:
                        tmpt_flame_board[x][yy] = 4
                        break

            for i in range(1, radius+1):
                yy = y + i
                if 0 <= yy < 11:
                    if (tmpt_board[x][yy] in (1,5)):
                        break
                    elif  tmpt_board[x][yy] == 3:
                        if (x, yy) not in visited:
                            cross_fire(x, yy, int(tmpt_bombs[x][yy]-1))
                        break
                    elif tmpt_board[x][yy] in (0,4):
                        tmpt_flame_board[x][yy] = 4
                    else:
                        tmpt_flame_board[x][yy] = 4
                        break

        #引爆炸弹
        for i in range(11):
            for j in range(11):
                if tmpt_bombs_life_time[i][j] == 1:
                    cross_fire(i, j, int(tmpt_bombs[i][j]-1))

        #炸弹时间-1
        for i in range(11):
            for j in range(11):
                if tmpt_bombs_life_time[i][j] != 0:
                    tmpt_bombs_life_time[i][j] -= 1

        #根据flame_board更新其他数组
        for i in range(11):
            for j in range(11):
                if tmpt_flame_board[i][j] != 0:
                    tmpt_flame_board[i][j] -= 1
                if tmpt_flame_board[i][j] != 0:
                    tmpt_board[i][j] = 4
                    tmpt_bombs[i][j] = 0
                    tmpt_bombs_life_time[i][j] = 0

        # 炸弹放置更新
        for x1 in range(11):
            for y1 in range(11):
                if tmpt_bombs[x1, y1] != 0 and tmpt_board[x1, y1] == 0:
                    tmpt_board[x1, y1] = 3

        next_state.flame_board = tmpt_flame_board
        next_state.set_current_board(tmpt_board)
        next_state.set_current_bombs(tmpt_bombs)
        next_state.set_current_move(actions[my_id - 10])
        next_state.set_target_enemy(self.target_enemy)
        # new_state.set_available_actions(self.actions)
        return next_state

    def __hash__(self):
        return hash(self.move)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False


class Node(object):
    """
    蒙特卡罗树搜索的树结构的Node，包含了父节self.enemies点和直接点等信息，还有用于计算UCB的遍历次数和quality值，还有游戏选择这个Node的State。
    """

    def __init__(self):
        self.parent = None
        self.children = []
        self.visit_times = 1
        self.quality_value = 0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        if len(self.children) == len(self.get_state().get_available_action()):
            return True
        else:
            return False

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def __repr__(self):
        return "Node: {}, Q/N: {}/{}, state: {}".format(
            hash(self), self.quality_value, self.visit_times, self.state)


def selection(node):
    # print("selection")
    if node.is_all_expand():
        sub_node = best_child(node)
    else:
        # Return the new sub node
        sub_node = expand(node)
    return sub_node


def simulation(node):
    """
    模拟三步写在state_stimulate里面
    """
    # print("stimulation")
    start = time.time()
    current_state = node.get_state()

    index = 0
    NUMBER = random.randint(0, 5)
    while index < NUMBER and current_state.is_terminal() == 3:
        # print("index",index)
        index += 1

        current_state = current_state.get_next_state_with_random_choice()
    final_state_reward = current_state.compute_reward()#越长期的模拟越可能不准确，因此得分要加权?
    # print("simulation: ", 1000 * (time.time() - start))
    return final_state_reward


def expand(node):
    """
    输入一个节点，在该节点上拓展一个新的节点，使用random方法执行Action，返回新增的节点。注意，需要保证新增的节点与其他节点Action不同。
    """
    # print("expand")
    tried_sub_node_states_move = [
        sub_node.get_state().get_current_move() for sub_node in node.get_children()
    ]

    new_state = node.get_state().get_next_state_with_random_choice()

    # Check until get the new state which has the different action from others
    while new_state.get_current_move() in tried_sub_node_states_move:
        new_state = node.get_state().get_next_state_with_random_choice()

    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    sub_node.set_parent(node)
    # for sub_node in node.get_children():
    #     print("sub_node", sub_node.get_state().get_current_move(), end = " ")
    # print(" ")
    return sub_node


def best_child(node):
    def node_score(node):  # 目前是UCB选取模式
        my_score = node.get_quality_value()
        my_visit_times = node.get_visit_times()
        my_parent_visit_times = node.get_parent().get_visit_times()
        score = my_score / my_visit_times + 60 * math.sqrt((2 * math.log(my_parent_visit_times) / my_visit_times))
        return score

    """
    得分平均值策略
    """
    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None

    # Travel all sub nodes to find the best one
    for sub_node in node.get_children():
        score = node_score(sub_node)
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node


def best_child_with_score(node):
    def node_score_with_score(node):  #
        my_score = node.get_quality_value()
        my_visit_times = node.get_visit_times()
        my_parent_visit_times = node.get_parent().get_visit_times()
        score = my_score / my_visit_times
        return score

    """
    得分平均值策略
    """
    # TODO: Use the min float value
    best_score = -sys.maxsize
    best_sub_node = None

    # Travel all sub nodes to find the best one
    for sub_node in node.get_children():
        score = node_score_with_score(sub_node)
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    return best_sub_node


def backup(node, reward):
    """
    模拟的得分加到节点上
    """
    # 模拟次数+1
    node.visit_times_add_one()
    # 更新总分
    node.quality_value_add_n(reward)
    node.get_parent().visit_times_add_one()


def MCTS_search(obs, mode, enemy=None):
    # Create the initialized state and initialized node
    # print(obs)
    init_state = State(obs, mode)
    init_state.set_target_enemy(enemy)
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node

    # Set the rounds to play

    computation_budget = 40 # 假设每次模拟1000次
    for i in range(computation_budget):
        # 1. Find the best node to expand
        expand_node = selection(current_node)

        # 2. Random run to add node and get reward
        reward = simulation(expand_node)

        # 3. Update all passing nodes with reward
        backup(expand_node, reward)

    tmpt_score = -sys.maxsize
    tmpt_node = None
    for sub_node in current_node.get_children():
        print(sub_node.get_state().get_current_move(), " : ", end = "")
        score = (sub_node.get_quality_value() / sub_node.get_visit_times())
        if score > tmpt_score:
            tmpt_score = score
            tmpt_node = sub_node
        print("value: ", sub_node.get_quality_value(), end = "")
        print("  visit_times:  ", sub_node.get_visit_times(), end = "")
        print("  score: ", score)

    print(" ")
    # time.sleep(2)

    return tmpt_node.get_state().get_current_move()
