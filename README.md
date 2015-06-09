wesnoth-mapgen
========================
Simple random map generator for the game Battle for Wesnoth, which can only create sea island maps for 2 players with their castles connected with one road.


Requirements:
-------------
* [__matplotlib__](http://matplotlib.org/)
* [__noise__](https://github.com/caseman/noise)
* [__numpy__](http://www.numpy.org/)
* [__pyglet__](https://bitbucket.org/pyglet/pyglet/wiki/Home)
* [__scipy__](http://www.scipy.org/)

---

Output.
-------
  Default output file is named `2p_Test.map`.
  To place generated map into the game you need to place it into `../GAME_FOLDER/data/multiplayer/maps` and [2p_Test.cfg](../master/2p_Test.cfg) (scenario file) into `../GAME_FOLDER/data/multiplayer/scenarios`.

Usage:
------
  To run a map generator use this command with different options. 
  
  For beginning try asking for help:

        python __init__.py -h
