#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import getopt
import hashlib
import sys

from map_gen import Map


def usage():
    print ""
    print "  Valid arguments are:"
    print ""
    print "    --debug - show this message"
    print "    --help  - show this message"
    print "    --ie    - island effect, default \"0.39\""
    print "    --out   - output file, default \"2p_Test.map\""
    print "    --seed  - seed for RNG"
    print "    --vpt   - villages pet thousand hexes, default \"20\""
    print "    -d      - same as --debug"
    print "    -h      - same as --help"
    print "    -i      - same as --ie"
    print "    -o      - same as --out"
    print "    -s      - same as --seed"
    print "    -v      - same as --vpt"
    print "    -x      - horizontal size of generated map, bigger than 8, default \"64\""
    print "    -y      - vertical size of generated map, bigger than 8, default \"64\""
    print ""


def main(argv):
    debug = False
    island_e = 0.39
    out = "2p_Test.map"
    seed = None
    vpt = 20
    x_size = 64
    y_size = 64
    try:
        # _ was args
        opts, _ = getopt.getopt(
            argv,
            "dhi:o:s:v:x:y:",
            ["help", "debug", "ie=", "out=", "seed=", "vpt="]
        )
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            usage()
            sys.exit()
        elif opt in ('-d', "--debug"):
            debug = True
        elif opt in ('-i', "--ie"):
            island_e = float(arg)
        elif opt in ('-o', "--out"):
            out = arg
        elif opt in ('-s', "--seed"):
            seed = int(hashlib.sha1(arg).hexdigest(), 16) % 4294967295
        elif opt in ('-v', "--vpt"):
            vpt = int(arg)
        elif opt == '-x':
            if int(arg) > 7:
                x_size = int(arg)
        elif opt == '-y':
            if int(arg) > 7:
                y_size = int(arg)

    # source = "".join(args)

    x_size += 2
    y_size += 2

    Map(x_size, y_size, out, vpt, island_e, debug, seed)


if __name__ == '__main__':
    main(sys.argv[1:])
