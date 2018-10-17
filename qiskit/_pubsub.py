# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Message broker for the Publisher / Subscriber mechanism
"""

import functools


class _Broker(object):

    _instance = None
    _subscribers = {}

    def __new__(cls):
        if _Broker._instance is None:
            _Broker._instance = object.__new__(cls)
        return _Broker._instance

    class _Subscription:
        def __init__(self, event, callback):
            self.event = event
            self.callback = callback

        def __eq__(self, other):
            """Overrides the default implementation"""
            if isinstance(other, self.__class__):
                return self.event == other.event and self.callback == other.callback
            return False

    def subscribe(self, event, callback):
        """ Subscribes to an event, so when it's emitted all the callbacks subscribed to it will be executed
        We are not allowing double registration

        :param event: The event to subscribed
        :param callback: The callback that will be executed when a event is emitted
        """
        if event not in self._subscribers:
            self._subscribers[event] = []

        new_subscription = self._Subscription(event, callback)
        if new_subscription in self._subscribers[event]:
            # We are not allowing double subscription
            return False

        self._subscribers[event].append(new_subscription)
        return True

    def dispatch(self, event, *args, **kwargs):
        """ Emits an event if there are any subscribers

        :param event: The event to be emitted
        :param args: Arguments linked with the event
        :param kwargs: Named arguments linked with the event
        """
        # No event, no subscribers.
        if event not in self._subscribers:
            return

        for subscriber in self._subscribers[event]:
            subscriber.callback(*args, **kwargs)

    def unsubscribe(self, event, callback):
        """ Unsubscribe the specific callback to the event
        :param event: The event to unsubscribe
        :param callback: The callback that won't be executed anymore
        :return True: if we have successfully unsubscribed to the event
        :return False: if there's no callback previously registered
        """

        try:
            self._subscribers[event].remove(self._Subscription(event, callback))
        except KeyError:
            return False

        return True


class Publisher(object):
    def __init__(self):
        self._broker = _Broker()

    def publish(self, event, *args, **kwargs):
        return self._broker.dispatch(event, *args, **kwargs)


class Subscriber(object):
    def __init__(self):
        self._broker = _Broker()

    def subscribe(self, event, callback):
        return self._broker.subscribe(event, callback)

    def unsubscribe(self, event, callback):
        return self._broker.unsubscribe(event, callback)
