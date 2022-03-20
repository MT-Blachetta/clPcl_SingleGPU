"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'mnist','cifar-10', 'stl-10', 'cifar-20', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '/space/blachetta/cifar-10/'
        
        elif database == 'cifar-20':
            return '/space/blachetta/cifar-20/'

        elif database == 'stl-10':
            return '/space/blachetta/data/'
        
        elif database == 'mnist':
            return '/space/blachetta/MNIST/'
        
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '/space/blachetta/imagenet/'
        
        else:
            raise NotImplementedError
