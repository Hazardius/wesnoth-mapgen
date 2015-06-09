#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import heapq
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy

from noise import pnoise2

# Carefull - in whole file:
# x - vertical
# y - horizontal

class Map(object):


    def __init__(self, size_x, size_y, dest_file="2p_Test.map", vpt=20, ie=0.39, debug=False, seed=None):
        self.debug = debug
        self.seed = seed
        self.map_size_x = size_y
        self.map_size_y = size_x
        self.villages_per_thousand = vpt
        self.island_effect = ie
        self.number_of_players = 2
        self.max_castle_fails = vpt * vpt
        self.hex_map = []
        self.altitude = []
        self.moisture = []
        if self.debug:
            print "generate()"
        self.generate()
        if self.debug:
            print "hexify()"
        self.hexify()
        if self.debug:
            print "populate_with_vilages()"
        self.populate_with_vilages()
        if self.debug:
            print "create_castles()"
        self.create_castles()
        if self.debug:
            self.create_image_from_map()
        self.save_to_file(dest_file)


    def save_to_file(self, dest_file):
        hex_file = codecs.open(dest_file, "w", "utf-8")
        hex_file.write("border_size=1\nusage=map\n\n")
        for x in range(self.map_size_x):
            for y in range(self.map_size_y):
                hex_file.write(self.hex_map[x][y])
                if y != self.map_size_y - 1:
                    hex_file.write(", ")
            if x != self.map_size_x - 1:
                hex_file.write("\n")


    def create_image_from_map(self):

        def color_from_hex_field(coord_x, coord_y):
            terrain = self.hex_map[coord_x][coord_y].split('^')
            if (len(terrain) > 1):
                if terrain[1][0] == 'V':
                    terrain = [terrain[0]]
                elif terrain[0] == "Gg":
                    return [0.0, 102.0, 0.0]
                else:
                    return [51.0, 102.0, 0.0]

            if (len(terrain) == 1):
                terrain = terrain[0]
                if terrain == "Ms":
                    return [255.0, 255.0, 255.0]
                elif terrain == "Mm":
                    return [ 51.0,  25.0,   0.0]
                elif terrain == "Md":
                    return [102.0,  51.0,   0.0]
                elif terrain == "Hd":
                    return [255.0, 255.0, 102.0]
                elif terrain == "Dd":
                    return [255.0, 255.0, 153.0]
                elif terrain == "Ds":
                    return [255.0, 255.0, 204.0]
                elif terrain == "Ww":
                    return [  0.0, 153.0, 153.0]
                elif terrain == "Ss":
                    return [ 25.0,  51.0,   0.0]
                elif terrain == "Gg":
                    return [153.0, 255.0,  51.0]
                elif terrain == "Gs":
                    return [178.0, 255.0, 102.0]
                elif terrain == "Gd":
                    return [204.0, 255.0, 153.0]
                elif terrain == "Hh":
                    return [153.0, 153.0,   0.0]
                elif terrain == "Hhd":
                    return [204.0, 204.0,   0.0]
                elif terrain == "Rr":
                    return [128.0, 128.0, 128.0]
                elif terrain == "Ch":
                    return [ 51.0,   0.0,  51.0]
                else:
                    return [102.0,   0.0, 102.0]

        bitmap = []
        for x in range(self.map_size_x):
            bitmap_row = []
            bitmap_row_weld = []
            for y in range(self.map_size_y):
                color = [canal / 255.0 for canal in color_from_hex_field(x, y)]
                if color == None:
                    print self.hex_map[x][y]
                bitmap_row.append(color)
                bitmap_row.append(color)

                if x + 1 < self.map_size_x:
                    if y % 2 != 0:
                        color = [canal / 255.0 for canal in color_from_hex_field(x + 1, y)]
                    bitmap_row_weld.append(color)
                    bitmap_row_weld.append(color)

            bitmap.append(bitmap_row)
            if x + 1 < self.map_size_x:
                bitmap.append(bitmap_row_weld)

        # pylint: disable=E1103
        bitmap = np.array(bitmap, np.float32)
        plt.imshow(bitmap)
        plt.savefig('generated_map.png', bbox_inches=0)
        # pylint: enable=E1103


    def generate(self):
        # pylint: disable=E1103
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.seed(self.map_size_x + self.map_size_y + np.random.randint(0, 256))
        octaves_alt = np.random.randint(0, 16) + 1
        octaves_moi = np.random.randint(0, 16) + 1
        # pylint: enable=E1103
        freq_alt = 16.0 * octaves_alt
        freq_moi = 16.0 * octaves_moi
        bitmap_data = []
        for x in range(self.map_size_x):
            bitmap_data_row = []
            for y in range(self.map_size_y):
                bit = []
                bit.append(0.0)
                bit.append(pnoise2(x / freq_alt, y / freq_alt, octaves_alt) * 0.5 + 0.5)
                bitmap_data_row.append(bit)
            bitmap_data.append(bitmap_data_row)

        # separately to get more different images of altitude and moisture
        for x in range(self.map_size_x):
            for y in range(self.map_size_y):
                bitmap_data[x][y].append(pnoise2(x / freq_moi, y / freq_moi, octaves_moi) * 0.5 + 0.5)

        bitmap_data = np.array(bitmap_data, np.float32)

        if self.debug:
            print "generate(): altitude and moisture maps created"

        bitmap_data = self.equalizeTerrain(bitmap_data)

        if self.debug:
            print "generate(): first equalization of terrain completed"

        for x in range(self.map_size_x):
            for y in range(self.map_size_y):
                x_val = (x - self.map_size_x / 2.0)
                y_val = (y - self.map_size_y / 2.0)
                radiusSquared = (x_val * x_val + y_val * y_val) * 1.0
                if abs(x_val) < abs(y_val):
                    temp_max = abs(y_val)
                else:
                    temp_max = abs(x_val)
                radiusSquared += pow(temp_max, 2.0)
                max_radiusSquared = self.map_size_x * self.map_size_y / 4.0
                radiusSquared /= max_radiusSquared
                radiusSquared += np.random.random() * 0.1
                radiusSquared *= self.island_effect
                island_multiplier = math.exp(-radiusSquared / 4.0) - radiusSquared
                island_multiplier = pow(island_multiplier, 3.0)
                c = bitmap_data[x][y]
                height = int(c[1] * 256)
                height = int(island_multiplier * height)
                if height < 0:
                    height = 0
                elif height > 255:
                    height = 255
                c[1] = (height / 256.0)
                bitmap_data[x][y] = c

        if self.debug:
            print "generate(): map changed to more island-like"

        bitmap_data = self.equalizeTerrain(bitmap_data)

        if self.debug:
            print "generate(): second equalization of terrain completed"

        self.altitude = []
        self.moisture = []
        for x in range(self.map_size_x):
            alt_row = []
            moi_row = []
            for y in range(self.map_size_y):
                c = bitmap_data[x][y]

                alt_row.append(c[1])
                moi_row.append(c[2])

            self.altitude.append(alt_row)
            self.moisture.append(moi_row)

        if self.debug:
            image_of_alt = []
            image_of_moi = []
            for x in range(self.map_size_x):
                im_alt_row = []
                im_moi_row = []
                for y in range(self.map_size_y):
                    c = bitmap_data[x][y]

                    alt = c[1]
                    if alt < 1.0/255.0:
                        color = [0.0, 0.0, 0.0]
                    elif alt < 0.5:
                        color = [2*alt, 1.0, 0.0]
                    else:
                        color = [1.0, 2*(1-alt), 0.0]

                    im_alt_row.append(color)

                    im_moi_row.append([0.0, 0.0, c[2]])

                image_of_alt.append(im_alt_row)
                image_of_moi.append(im_moi_row)

            # pylint: disable=E1103
            image_of_alt = np.array(image_of_alt, np.float32)
            plt.imshow(image_of_alt)
            plt.savefig('altitude.png', bbox_inches=0)
            image_of_moi = np.array(image_of_moi, np.float32)
            plt.imshow(image_of_moi)
            plt.savefig('moisture.png', bbox_inches=0)
            # pylint: enable=E1103


    def transformSpectrumOnBitmap(self, bitmap_data, red=None, green=None, blue=None):
        new_bitmap = []
        for x in range(self.map_size_x):
            new_bitmap_row = []
            for y in range(self.map_size_y):
                original_bit = bitmap_data[x][y]
                bit = []
                if red == None:
                    bit.append(original_bit[0])
                else:
                    index = int(original_bit[0] * 255)
                    bit.append(red[index] / 255.0)
                if green == None:
                    bit.append(original_bit[1])
                else:
                    index = int(original_bit[1] * 255)
                    bit.append(green[index] / 255.0)
                if blue == None:
                    bit.append(original_bit[2])
                else:
                    index = int(original_bit[2] * 255)
                    bit.append(blue[index] / 255.0)
                new_bitmap_row.append(bit)
            new_bitmap.append(new_bitmap_row)
        # pylint: disable=E1103
        return np.array(new_bitmap, np.float32)
        # pylint: enable=E1103


    def equalizeTerrain(self, bitmap_data):
        G = np.histogram(
            [[pixel[1] for pixel in row] for row in bitmap_data],
            256,
            range=(0.0, 1.0)
            )[0]
        B = np.histogram(
            [[pixel[2] for pixel in row] for row in bitmap_data],
            256,
            range=(0.0, 1.0)
            )[0]
        g = 0
        b = 0
        green = []
        blue = []
        cumsumG = 0.0
        cumsumB = 0.0
        for i in range(256):
            cumsumG += G[i]
            cumsumB += B[i]
            green.append(g * g / 255.0)
            blue.append(b * b / 255.0)
            while cumsumG > self.map_size_x * self.map_size_y * (g / 256.0) and g < 255:
                g += 1
            while cumsumB > self.map_size_x * self.map_size_y * (b / 256.0) and b < 255:
                b += 1
        return self.transformSpectrumOnBitmap(bitmap_data, None, green, blue)


    def hexify(self):
        self.hex_map = []

        def average_bit_points_to_hex(x, y):
            sum_altitude = 0
            sum_moisture = 0
            next_x = (x + 1 < self.map_size_x - 1)
            next_y = (y + 1 < self.map_size_y - 1)
            prev_y = (y - 1 > 0)
            if next_x:
                sum_altitude += self.altitude[x + 1][y]
                sum_moisture += self.moisture[x + 1][y]
                if prev_y:
                    sum_altitude += self.altitude[x + 1][y - 1]
                    sum_moisture += self.moisture[x + 1][y - 1]
                if next_y:
                    sum_altitude += self.altitude[x + 1][y + 1]
                    sum_moisture += self.moisture[x + 1][y + 1]
            sum_altitude += self.altitude[x][y]
            sum_moisture += self.moisture[x][y]
            if prev_y:
                sum_altitude += self.altitude[x][y - 1]
                sum_moisture += self.moisture[x][y - 1]
            if next_y:
                sum_altitude += self.altitude[x][y + 1]
                sum_moisture += self.moisture[x][y + 1]
            return (sum_altitude / 6.0), (sum_moisture / 6.0)

        for x in range(self.map_size_x):
            hex_map_row = []
            for y in range(self.map_size_y):
                if (y % 2 == 0) or (x == 0):
                    altitude = self.altitude[x][y]
                    moisture = self.moisture[x][y]
                else:
                    altitude, moisture = average_bit_points_to_hex(x, y)
                if altitude < 16/255.0:
                    hex_map_row.append("Ww")
                elif altitude < 32/255.0:
                    hex_map_row.append("Ds")
                elif altitude < 156/255.0:
                    if moisture < 16/255.0:
                        hex_map_row.append("Dd")
                    elif moisture < 53/255.0:
                        hex_map_row.append("Gd")
                    elif moisture < 90/255.0:
                        hex_map_row.append("Gs")
                    elif moisture < 128/255.0:
                        hex_map_row.append("Gg")
                    elif moisture < 192/255.0:
                        hex_map_row.append("Gg^Fms")
                    else:
                        hex_map_row.append("Ss")
                elif altitude < 192/255.0:
                    if moisture < 16/255.0:
                        hex_map_row.append("Hd")
                    elif moisture < 72/255.0:
                        hex_map_row.append("Hhd")
                    elif moisture < 192/255.0:
                        hex_map_row.append("Hh")
                    else:
                        hex_map_row.append("Hh^Fms")
                elif altitude < 248/255.0:
                    if moisture < 72/255.0:
                        hex_map_row.append("Md")
                    else:
                        hex_map_row.append("Mm")
                else:
                    hex_map_row.append("Ms")
            self.hex_map.append(hex_map_row)


    def create_village(self):
        # pylint: disable=E1103
        added = False
        x = np.random.randint(0, self.map_size_x - 4) + 2
        y = np.random.randint(0, self.map_size_y - 4) + 2
        # pylint: enable=E1103
        terrain = self.hex_map[x][y]
        if terrain in ("Dd", "Hd"):
            # pylint: disable=E1103
            if np.random.randint(0, 2) == 0:
            # pylint: enable=E1103
                self.hex_map[x][y] += "^Vda"
            else:
                self.hex_map[x][y] += "^Vdt"
            added = True
        elif terrain == "Ss":
            self.hex_map[x][y] += "^Vhs"
            added = True
        elif terrain in ("Gd", "Gs", "Gg"):
            # pylint: disable=E1103
            choice = np.random.randint(0, 9)
            # pylint: enable=E1103
            if choice == 0:
                self.hex_map[x][y] += "^Vct"
            elif choice == 1:
                self.hex_map[x][y] += "^Vo"
            elif choice == 2:
                self.hex_map[x][y] += "^Vh"
            elif choice == 3:
                self.hex_map[x][y] += "^Vhr"
            elif choice == 4:
                self.hex_map[x][y] += "^Vhc"
            elif choice == 5:
                self.hex_map[x][y] += "^Vwm"
            elif choice == 6:
                self.hex_map[x][y] += "^Vhcr"
            elif choice == 7:
                self.hex_map[x][y] += "^Vc"
            elif choice == 8:
                self.hex_map[x][y] += "^Vl"
            added = True
        elif terrain in ("Hhd", "Hh"):
            # pylint: disable=E1103
            choice = np.random.randint(0, 5)
            # pylint: enable=E1103
            if choice == 0:
                self.hex_map[x][y] += "^Vu"
            elif choice == 1:
                self.hex_map[x][y] += "^Vud"
            elif choice == 2:
                self.hex_map[x][y] += "^Vhh"
            elif choice == 3:
                self.hex_map[x][y] += "^Vhhr"
            elif choice == 4:
                self.hex_map[x][y] += "^Vd"
            added = True
        elif terrain == "Gg^Fms":
            self.hex_map[x][y] = "Gg^Ve"
            added = True
        if added:
            self.villages_coords.append([x, y])
        return added


    def populate_with_vilages(self):
        self.villages_coords = []
        limit_of_villages = (
            self.villages_per_thousand * self.map_size_x * self.map_size_y / 1000.0
        ) + 2.0
        while len(self.villages_coords) + 1 <= limit_of_villages:
            self.create_village()
            pass


    def create_castles(self):
        vil_count = len(self.villages_coords)
        fail = True
        fail_counter = 0
        distance = math.sqrt(pow(self.map_size_x, 2.0) + pow(self.map_size_y, 2.0)) / 2.5
        if distance < 5:
            distance = 5
        if self.debug:
            print "create_castles(): try to use existing villages"
        while fail:
            # pylint: disable=E1103
            v_1 = np.random.randint(0, vil_count)
            # pylint: enable=E1103
            v_2 = v_1
            while v_2 == v_1:
                # pylint: disable=E1103
                v_2 = np.random.randint(0, vil_count)
                # pylint: enable=E1103
            x_sqr = pow(self.villages_coords[v_1][0] - self.villages_coords[v_2][0], 2.0)
            y_sqr = pow(self.villages_coords[v_1][1] - self.villages_coords[v_2][1], 2.0)
            if math.ceil(math.sqrt(x_sqr + x_sqr)) > distance:
                self.create_road_between(self.villages_coords[v_1], self.villages_coords[v_2])
                self.create_castle(self.villages_coords[v_1], 1)
                self.create_castle(self.villages_coords[v_2], 2)
                fail = False
                break
            fail_counter += 1
            if fail_counter >= self.max_castle_fails:
                fail = False
        if fail_counter >= self.max_castle_fails:
            if self.debug:
                print "create_castles(): using villages failed"
                print "create_castles(): try to use any non-water field"
            fail = True
            fail_counter = 0
            zero_point = [0, 0]
            while fail:
                point_1 = [0, 0]
                point_2 = [0, 0]
                while self.hex_map[point_1[0]][point_1[1]] == "Ww" or point_1 == zero_point:
                    point_1 = []
                    point_1.append(np.random.randint(0, self.map_size_x - 4) + 2)
                    point_1.append(np.random.randint(0, self.map_size_y - 4) + 2)
                while self.hex_map[point_2[0]][point_2[1]] == "Ww" or point_2 == zero_point:
                    point_2 = []
                    point_2.append(np.random.randint(0, self.map_size_x - 4) + 2)
                    point_2.append(np.random.randint(0, self.map_size_y - 4) + 2)
                x_sqr = pow(point_1[0] - point_2[0], 2.0)
                y_sqr = pow(point_1[1] - point_2[1], 2.0)
                if  math.ceil(math.sqrt(x_sqr + x_sqr)) > distance:
                    self.create_road_between(point_1, point_2)
                    self.create_castle(point_1, 1)
                    self.create_castle(point_2, 2)
                    fail = False
                fail_counter += 1
                if fail_counter >= self.max_castle_fails:
                    distance -= 1.0
                    fail_counter = 0


    def create_road_between(self, center_of_castle_1, center_of_castle_2):
        castle_1_x = center_of_castle_1[0]
        castle_1_y = center_of_castle_1[1]
        castle_2_x = center_of_castle_2[0]
        castle_2_y = center_of_castle_2[1]

        class PriorityQueue(object):
            def __init__(self):
                self.elements = []
            
            def empty(self):
                return len(self.elements) == 0
            
            def put(self, item, priority):
                heapq.heappush(self.elements, (priority, item))
            
            def get(self):
                return heapq.heappop(self.elements)[1]

        def heuristic(a, b):
            (x1, y1) = a
            (x2, y2) = b
            return math.ceil(math.sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0)))

        def neighbors(current):
            x, y = current
            neighbors = []
            next_x = (x + 1 < self.map_size_x - 1)
            prev_x = (x - 1 > 0)
            next_y = (y + 1 < self.map_size_y - 1)
            prev_y = (y - 1 > 0)
            if next_x:
                neighbors.append((x + 1, y))
            if prev_x:
                neighbors.append((x - 1, y))
            if next_y:
                neighbors.append((x, y + 1))
            if prev_y:
                neighbors.append((x, y - 1))
            if y % 2 == 0:
                if next_x:
                    if prev_y:
                        neighbors.append((x + 1, y - 1))
                    if next_y:
                        neighbors.append((x + 1, y + 1))
            else:
                if prev_x:
                    if prev_y:
                        neighbors.append((x - 1, y - 1))
                    if next_y:
                        neighbors.append((x - 1, y + 1))
            return neighbors

        def cost(coords):
            x, y = coords
            terrain = self.hex_map[x][y].split("^V")
            if len(terrain) == 1:
                if terrain[0] in ("Gd", "Gs", "Gg"):
                    return 100
                if terrain[0] == "Ds":
                    return 125
                if terrain[0] == "Gg^Fms":
                    return 150
                if terrain[0] in "Ss":
                    return 175
                if terrain[0] == "Dd":
                    return 200
                if terrain[0] in ("Hhd", "Hh"):
                    return 300
                if terrain[0] == "Hh^Fms":
                    return 400
                if terrain[0] == "Hd":
                    return 600
                if terrain[0] in ("Md", "Mm"):
                    return 800
                if terrain[0] == "Ms":
                    return 1000
                if terrain[0] == "Ww":
                    return 1000000
            else:
                if terrain[0] in ("Gd", "Gs"):
                    return 50
                if terrain[0] == "Gg":
                    if (terrain[1] == "e"):
                        return 100
                    return 50
                if terrain[0] in "Ss":
                    return 125
                if terrain[0] == "Dd":
                    return 150
                if terrain[0] in ("Hhd", "Hh"):
                    return 250
                if terrain[0] == "Hd":
                    return 500

        def a_star_search(start, goal):
            frontier = PriorityQueue()
            frontier.put(start, 0)
            came_from = {}
            cost_so_far = {}
            came_from[start] = None
            cost_so_far[start] = 0
            
            while not frontier.empty():
                current = frontier.get()

                if current == goal:
                    break

                for next in neighbors(current):
                    new_cost = cost_so_far[current] + cost(next)
                    if next not in cost_so_far or new_cost < cost_so_far[next]:
                        cost_so_far[next] = new_cost
                        priority = new_cost + heuristic(goal, next)
                        frontier.put(next, priority)
                        came_from[next] = current
            
            return came_from, cost_so_far

        came_from, _ = a_star_search((castle_1_x, castle_1_y), (castle_2_x, castle_2_y))

        def change_terrain_to_road(hex_coords):
            x, y = hex_coords
            prev_terrain = self.hex_map[x][y].split("^")
            if len(prev_terrain) == 1:
                self.hex_map[x][y] = "Rr"
            elif (prev_terrain[1][0] == 'V'):
                self.hex_map[x][y] = "Rr^" + prev_terrain[1]
            else:
                self.hex_map[x][y] = "Rr"

        current_hex = (castle_2_x, castle_2_y)
        while current_hex != (castle_1_x, castle_1_y):
            change_terrain_to_road(current_hex)
            current_hex = came_from[current_hex]


    def create_castle(self, center_of_castle, player_nr):
        x = center_of_castle[0]
        y = center_of_castle[1]
        destroyed_villages = 0

        def check_if_destroying_village(x_val, y_val):
            prev = self.hex_map[x_val][y_val].split('^')
            if (len(prev) > 1 and prev[1][0] == 'V'):
                return 1
            return 0

        destroyed_villages += check_if_destroying_village(x, y)
        self.hex_map[x][y] = str(player_nr) + " Kh"
        destroyed_villages += check_if_destroying_village(x - 1, y)
        self.hex_map[x - 1][y] = "Ch"
        destroyed_villages += check_if_destroying_village(x + 1, y)
        self.hex_map[x + 1][y] = "Ch"
        destroyed_villages += check_if_destroying_village(x, y - 1)
        self.hex_map[x][y - 1] = "Ch"
        destroyed_villages += check_if_destroying_village(x, y + 1)
        self.hex_map[x][y + 1] = "Ch"
        if y % 2 == 0:
            destroyed_villages += check_if_destroying_village(x + 1, y - 1)
            self.hex_map[x + 1][y - 1] = "Ch"
            destroyed_villages += check_if_destroying_village(x + 1, y + 1)
            self.hex_map[x + 1][y + 1] = "Ch"
        else:
            destroyed_villages += check_if_destroying_village(x - 1, y - 1)
            self.hex_map[x - 1][y - 1] = "Ch"
            destroyed_villages += check_if_destroying_village(x - 1, y + 1)
            self.hex_map[x - 1][y + 1] = "Ch"

        while (destroyed_villages > 0):
            added = False
            while not added:
                added = self.create_village()
            destroyed_villages -= 1
