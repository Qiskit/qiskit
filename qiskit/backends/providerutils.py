# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Utilities for providers."""

import logging

logger = logging.getLogger(__name__)


def filter_backends(backends, filters=None, **kwargs):
    """Return the backends matching the specified filtering.

    Filter the `backends` list by their `configuration` or `status`
    attributes, or from a boolean callable. The criteria for filtering can
    be specified via `**kwargs` or as a callable via `filters`, and the
    backends must fulfill all specified conditions.

    Args:
        backends (list[BaseBackend]): list of backends.
        filters (callable): filtering conditions as a callable.
        **kwargs (dict): dict of criteria.

    Returns:
        list[BaseBackend]: a list of backend instances matching the
            conditions.
    """
    def _match_all(dict_, criteria):
        """Return True if all items in criteria matches items in dict_."""
        return all(dict_.get(key_) == value_ for
                   key_, value_ in criteria.items())

    # Inspect the backends to decide which filters belong to
    # backend.configuration and which ones to backend.status, as it does
    # not involve querying the API.
    configuration_filters = {}
    status_filters = {}
    for key, value in kwargs.items():
        if all(key in backend.configuration() for backend in backends):
            configuration_filters[key] = kwargs[key]
        else:
            status_filters[key] = kwargs[key]

    # 1. Apply backend.configuration filtering.
    if configuration_filters:
        backends = [b for b in backends if
                    _match_all(b.configuration(), configuration_filters)]

    # 2. Apply backend.status filtering (it involves one API call for
    # each backend).
    if status_filters:
        backends = [b for b in backends if
                    _match_all(b.status(), status_filters)]

    # 3. Apply acceptor filter.
    backends = list(filter(filters, backends))

    return backends


def resolve_backend_name(name, backends, grouped, deprecated, aliased):
    """Resolve backend name from a possible short group name, a deprecated name,
    or an alias.

    A group will be resolved in order of member priorities, depending on
    availability.

    Args:
        name (str): name of backend to resolve
        backends (list[BaseBackend]): list of available backends.
        grouped (dict[str: list[str]]): dict of grouped names.
        deprecated (dict[str: str]): dict of deprecated names.
        aliased (dict[str: list[str]]): dict of aliased names.

    Returns:
        list (str): list of name of resolved backends

    Raises:
        LookupError: if name cannot be resolved through regular available
            names, nor groups, nor deprecated, nor alias names.
    """
    resolved_names = []
    available = [backend.name() for backend in backends]

    if name in available:
        resolved_names = [name]
    elif name in grouped:
        available_members = [b for b in grouped[name] if b in available]
        if available_members:
            resolved_names = available_members
    elif name in deprecated:
        resolved_names = [deprecated[name]]
        logger.warning('WARNING: %s is deprecated. Use %s.', name, resolved_names[0])
    elif name in aliased:
        resolved_names = [aliased[name]]

    # Prune unavailable resolved names.
    resolved_names = [resolved_name for resolved_name in resolved_names
                      if resolved_name in available]

    if not resolved_names:
        raise LookupError('backend "{}" not found.'.format(name))

    return resolved_names
