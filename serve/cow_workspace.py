"""Read-only message projection over canonical COW snapshots and receipts.

This module has no SQLite tables, branch heads, native model calls, or writes.
Message text is reconstructed only from committed presentation spans and the
loaded tokenizer's token-byte interface.
"""

from __future__ import annotations

import codecs
from collections import defaultdict
import hashlib
import json
import re

from context_time_travel import IntegrityError, NotFound
from context_time_travel.input_spans import InputHistory, history_lineage, history_segment, generated_text


class _CatalogHeadStore:
    """Read-only Store view that substitutes one receipt-bound historical head."""

    def __init__(self, store, catalog, head):
        self._store = store
        self._catalog = catalog
        self._head = dict(head)

    def __getattr__(self, name):
        return getattr(self._store, name)

    def presentation_catalog(self, namespace_id):
        if namespace_id != self._catalog["namespace"]["namespace_id"]:
            raise NotFound("Namespace is not in the captured graph catalog")
        return {**self._catalog, "heads": [dict(self._head)]}


class CowWorkspace:
    """Project namespace-scoped Store receipts into ordinary chat-shaped data."""

    def __init__(self, store, tokenizer, *, root_mode_resolver=None):
        if not callable(getattr(store, "presentation_catalog", None)):
            raise TypeError("CowWorkspace requires Store.presentation_catalog")
        if not callable(getattr(tokenizer, "token_bytes", None)):
            raise TypeError("CowWorkspace requires tokenizer.token_bytes")
        self.store = store
        self.tokenizer = tokenizer
        if root_mode_resolver is not None and not callable(root_mode_resolver):
            raise TypeError("root_mode_resolver must be callable")
        self.root_mode_resolver = root_mode_resolver

    def _root_mode(self, catalog, namespace_id, root_id, root_record):
        if not callable(self.root_mode_resolver):
            return "unsupported"
        presentations = [event["presentation"] for event in catalog["events"]
            if event.get("result", {}).get("tip") == root_id
            and isinstance(event.get("presentation"), dict)
            and event["presentation"].get("kind") == "root"]
        return self.root_mode_resolver(namespace_id, root_id,
            InputHistory(root_record['inputs']) if root_record['schema']==2 else tuple(root_record['inputs']),
            root_record["controller"], presentations)

    def list_chats(self):
        """List presentation namespaces, including empty chats, without writes."""
        chats = []
        with self.store._locked():
            for namespace in self.store.namespaces():
                catalog = self.store.presentation_catalog(namespace.namespace_id)
                counts = [len(self._message_ids(catalog, head)) for head in catalog["heads"]]
                chats.append({
                    "id": namespace.namespace_id,
                    "chat_id": namespace.namespace_id,
                    "title": namespace.title,
                    "created_at": namespace.created_at_ns / 1_000_000_000,
                    "branch_count": len(catalog["heads"]),
                    "max_saved_messages": max(counts, default=0),
                })
        chats.sort(key=lambda chat: (-chat["created_at"], chat["id"]))
        return chats

    def list_branches(self, chat_id):
        """Return current branch heads for exactly one immutable namespace."""
        branches = []
        with self.store._locked():
            catalog = self.store.presentation_catalog(chat_id)
            names = self._branch_display(catalog)
            for head in sorted(catalog["heads"], key=lambda item: (
                    names[item["name"]]["creation_order"], item["name"])):
                controller = self.store.get_snapshot(head["tip"])["controller"]
                branches.append({
                    "id": head["name"],
                    "chat_id": chat_id,
                    "name": names[head["name"]]["display_name"],
                    "creation_order": names[head["name"]]["creation_order"],
                    "revision": head["revision"],
                    "tip": head["tip"],
                    "root": head["root"],
                    "saved_messages": len(self._message_ids(catalog, head)),
                    "controller_status": controller.get("status"),
                    "pending_controller": controller.get("status") == "active",
                    "pending_request": None,
                })
        return branches

    def history(self, branch_id):
        """Return one branch's durable messages, moves, savepoints and controller flags."""
        with self.store._locked():
            namespace_id = self.store.branch_namespace(branch_id)
            if namespace_id is None:
                raise NotFound("Branch is not bound to a presentation namespace")
            catalog = self.store.presentation_catalog(namespace_id)
            branch_name = self._branch_display(catalog).get(branch_id, {}).get("display_name", branch_id)
            head = next((item for item in catalog["heads"] if item["name"] == branch_id), None)
            if head is None:
                raise NotFound("Branch is not in the namespace catalog")
            root_id, token_ids, path, positions = self._canonical_path(head)
            if root_id != head["root"]:
                raise IntegrityError("Selected COW branch lineage differs from its recorded root")
            ancestry = {identity for identity, _record in path}
            history_messages = self._project_messages(catalog, head, ancestry, token_ids, path, positions)
            saved_points = self._saved_points(catalog, head, ancestry)
            controller = path[-1][1]["controller"]
            pending_hex = controller.get("output_pending_hex", "")
            try:
                pending_utf8 = bool(bytes.fromhex(pending_hex))
            except (TypeError, ValueError) as error:
                raise IntegrityError("Controller has invalid pending UTF-8 bytes") from error
            status = controller.get("status")
            initial_root_mode = (self._root_mode(catalog, namespace_id, root_id, path[0][1])
                                 if head["tip"] == root_id else "unsupported")
            moves = []
            for event in catalog["events"]:
                if event["branch"] != branch_id or event["type"] != "move":
                    continue
                source_event = event["event"]
                moves.append({
                    "branch_id": branch_id,
                    "from_node": source_event["from"],
                    "to_node": source_event["to"],
                    "revision": source_event["to_revision"],
                    "operation_id": event["operation_id"],
                })
            result = {
                "id": branch_id,
                "chat_id": namespace_id,
                "name": branch_name,
                "revision": head["revision"],
                "tip": head["tip"],
                "root_id": root_id,
                "root": root_id,
                # Canonical model-input cells: one per text token or image row.
                # These counts neither admit computation nor describe residency.
                "root_input_cells": (len(InputHistory(path[0][1]["inputs"]))
                                     if path[0][1]["schema"] == 2 else len(path[0][1]["inputs"])),
                "tip_input_cells": len(token_ids),
                "messages": history_messages,
                "moves": moves,
                "saved_points": saved_points,
                "controller_status": status,
                "pending_controller": {
                    "active": status == "active",
                    "status": status,
                    "accepted_in_scope": controller.get("accepted_in_scope"),
                    "pending_utf8": pending_utf8,
                },
                "pending_request": None,
                "initial_root_mode": initial_root_mode,
                "execution_mode": "cow-store-presentation-projection",
            }
            if isinstance(token_ids,InputHistory):
                position=0;images=[]
                for span in token_ids.spans:
                    if span['kind']=='image':
                        images.append(dict(span,start=position));position+=span['rows']
                    else:position+=len(span['tokens'])
                # These are exact canonical placements for the branch. Root
                # annotations have no per-message cell offsets: do not guess
                # image-to-message ownership by parsing placeholder text.
                result['image_inputs']=images
            return result

    def graph(self, namespace_id, *, include_previous=False, preview_chars=160):
        """Build a transient message-version graph from one namespace snapshot.

        ``tip`` is a graph node ID; ``snapshot`` is the actual COW head. A node
        only carries an ``action_snapshot`` when its publication ended at a
        real canonical snapshot boundary. Assistant resumes create immutable
        span versions at the same logical message depth.
        """
        if type(include_previous) is not bool:
            raise ValueError("include_previous must be boolean")
        if type(preview_chars) is not int or preview_chars < 0:
            raise ValueError("preview_chars must be a non-negative integer")
        with self.store._locked():
            catalog = self.store.presentation_catalog(namespace_id)
            branch_names = self._branch_display(catalog)
            anchor = f"chat:{namespace_id}"
            nodes = {anchor: {
                "id": anchor, "parent_id": None, "depth": 0, "role": "root",
                "preview": "", "message_id": None, "message_increment": 0,
                "current_branch_count": 0, "action_snapshot": None,
                    "boundary_available": False, "shared_partial": False,
                    "content_sha256": hashlib.sha256(b"").hexdigest(),
            }}
            branches = []
            # Presentation event ordinals are the durable creation sequence.
            # Branch names are user chosen and must not be used as chronology.
            creation_order = {name: record["creation_order"]
                              for name, record in branch_names.items()}
            branch_paths = {}
            logical_paths = {}
            for head in catalog["heads"]:
                root_id, token_ids, path, positions = self._canonical_path(head)
                if root_id != head["root"]:
                    raise IntegrityError("Catalog branch root differs from canonical lineage")
                root_mode = self._root_mode(catalog, namespace_id, root_id, path[0][1])
                ancestry = {identity for identity, _record in path}
                messages = self._project_messages(catalog, head, ancestry, token_ids, path, positions)
                records = {identity: record for identity, record in path}
                by_message = {message["id"]: message for message in messages}
                # Convert only durable annotations selected by the validated
                # provenance resolver. Text is never an identity key.
                ordered = list(self._events_on_ancestry(catalog, ancestry, head))
                message_descriptors = {}
                assistant_segments = defaultdict(list)
                for event in ordered:
                    presentation = event["presentation"]
                    if presentation is None:
                        continue
                    kind = presentation["kind"]
                    for index, source in enumerate(presentation["messages"]):
                        if kind not in ("root", "initial_turn", "user_turn"):
                            continue
                        span = presentation.get("input_span")
                        start = span["start"] if span else None
                        end = start + span["count"] if span else None
                        key = source["id"]
                        message_descriptors.setdefault(key, {
                            "id": key, "role": source["role"], "text": source["content"],
                            "start": start, "end": end,
                            "order": (start if start is not None else 0, 0,
                                      event["ordinal"], index),
                            "snapshot": event["result"]["tip"] if kind != "root" else None,
                            "root_snapshot": event["result"]["tip"] if kind == "root" else None,
                        })
                    output = presentation.get("output_span")
                    if not isinstance(output, dict):
                        continue
                    start = output["start"]
                    end = start + output["count"]
                    result_tip = event["result"]["tip"]
                    source_tip = presentation.get("source", {}).get("snapshot")
                    source_record = records.get(source_tip)
                    result_record = records.get(result_tip)
                    if source_record is None or result_record is None:
                        raise IntegrityError("Assistant graph span is outside selected canonical ancestry")
                    assistant_segments[output["message_id"]].append({
                        "start": start, "end": end, "tokens": generated_text(history_segment(token_ids,start,end)),
                        "kind": presentation["kind"], "result_tip": result_tip,
                        "source_controller": source_record["controller"],
                        "result_controller": result_record["controller"],
                        "order": (event["ordinal"], event["branch"], event["operation_id"]),
                    })

                inputs = []
                for descriptor in message_descriptors.values():
                    projected = by_message.get(descriptor["id"])
                    if projected is None or projected["role"] != descriptor["role"]:
                        raise IntegrityError("Graph input annotation differs from selected message projection")
                    if projected["text"] != descriptor["text"]:
                        raise IntegrityError("One graph message ID has conflicting input text")
                    descriptor["text"] = projected["text"]
                    descriptor["boundary"] = projected["node_id"]
                    inputs.append(descriptor)
                for message_id, segments in assistant_segments.items():
                    projected = by_message.get(message_id)
                    if projected is None or projected["role"] != "assistant":
                        raise IntegrityError("Graph assistant annotation differs from selected message projection")
                    segments.sort(key=lambda segment: (segment["start"], segment["end"], segment["order"]))
                    message_descriptors[message_id] = {
                        "id": message_id, "role": "assistant", "text": projected["text"],
                        "start": projected["token_start"],
                        "end": projected["token_start"] + projected["token_count"],
                        "order": (projected["token_start"], 1, *segments[0]["order"]),
                        "segments": segments, "generated": True,
                        "boundaries": projected["snapshot_boundaries"],
                    }
                inputs.sort(key=lambda item: item["order"])
                assistants = [value for value in message_descriptors.values()
                              if value.get("generated")]
                assistants.sort(key=lambda item: item["order"])
                descriptors = sorted(inputs + assistants, key=lambda item: item["order"])

                root_node_id = self._graph_node_id(["root-snapshot", namespace_id, root_id])
                root_node = self._graph_node(root_node_id, anchor, 0,
                    None, "root", "", 0, root_id, "")
                root_node["initial_root_mode"] = root_mode
                self._insert_graph_node(nodes, root_node)
                current_parent = root_node_id
                depth = 0
                branch_node_path = [anchor, root_node_id]
                logical_path = []
                seen_messages = set()
                for descriptor in descriptors:
                    message_id = descriptor["id"]
                    projected = by_message[message_id]
                    if message_id in seen_messages:
                        raise IntegrityError("Presentation graph repeats one message identity on a branch")
                    seen_messages.add(message_id)
                    if not descriptor.get("generated"):
                        identity_material = [namespace_id, message_id, descriptor["start"],
                                             descriptor["end"], current_parent]
                        node_id = self._graph_node_id(identity_material)
                        depth += 1
                        node = self._graph_node(node_id, current_parent, depth, message_id,
                            descriptor["role"], descriptor["text"][:preview_chars], 1,
                            descriptor.get("boundary"), descriptor["text"])
                        self._insert_graph_node(nodes, node)
                        current_parent = node_id
                        branch_node_path.append(node_id)
                        logical_path.append(node_id)
                        continue

                    segments = descriptor["segments"]
                    cumulative = []
                    assistant_parent = current_parent
                    versions = []
                    emitted = set()
                    for segment in segments:
                        cumulative.append(segment)
                        endpoint = segment["end"]
                        key = (endpoint, segment["result_tip"])
                        if key in emitted:
                            continue
                        emitted.add(key)
                        cumulative_text = self._assistant_prefix_text(message_id, descriptor,
                            cumulative, endpoint, final=(endpoint == descriptor["end"]))
                        identity_material = [namespace_id, message_id,
                            segments[0]["start"], endpoint,
                            hashlib.sha256(json.dumps(generated_text(history_segment(token_ids,segments[0]['start'],endpoint)),
                                separators=(",", ":")).encode()).hexdigest(), assistant_parent]
                        node_id = self._graph_node_id(identity_material)
                        increment = 1 if not versions else 0
                        if increment:
                            depth += 1
                        node = self._graph_node(node_id, assistant_parent, depth, message_id,
                            "assistant", cumulative_text[:preview_chars], increment,
                            segment["result_tip"], cumulative_text)
                        self._insert_graph_node(nodes, node)
                        versions.append(node_id)
                        assistant_parent = node_id
                        branch_node_path.append(node_id)
                    if not versions:
                        # A zero-token answer still has a durable message record.
                        identity_material = [namespace_id, message_id, descriptor["start"],
                                             descriptor["end"], current_parent]
                        node_id = self._graph_node_id(identity_material)
                        depth += 1
                        self._insert_graph_node(nodes, self._graph_node(node_id, current_parent,
                            depth, message_id, "assistant", descriptor["text"][:preview_chars],
                            1, projected["node_id"], descriptor["text"]))
                        versions.append(node_id)
                        branch_node_path.append(node_id)
                    current_parent = versions[-1]
                    logical_path.append(current_parent)

                tip = current_parent
                controller = path[-1][1]["controller"]
                branches.append({
                    "id": head["name"],
                    "name": branch_names[head["name"]]["display_name"],
                    "creation_order": creation_order.get(head["name"], -1),
                    "revision": head["revision"], "tip": tip, "snapshot": head["tip"],
                    "root": head["root"], "saved_messages": len(messages),
                    "pending_request": None, "pending_controller": controller.get("status") == "active",
                    "root_mode": root_mode,
                })
                branch_node_path = tuple(dict.fromkeys(branch_node_path))
                branch_node_path_by_name = branch_node_path
                branch_paths[head["name"]] = branch_node_path_by_name
                logical_paths[head["name"]] = tuple(logical_path)

            previous_tips = []
            if include_previous:
                seen_previous = set()
                for event in catalog["events"]:
                    if event["type"] != "move":
                        continue
                    move = event["event"]
                    old_tip = move.get("from")
                    old_revision = move.get("from_revision")
                    if (not isinstance(old_tip, str) or type(old_revision) is not int
                            or old_revision < 0):
                        raise IntegrityError("Previous branch tip receipt is malformed")
                    identity = (event["branch"], old_revision, old_tip)
                    if identity in seen_previous:
                        continue
                    seen_previous.add(identity)
                    historical_head = {
                        "name": event["branch"], "revision": old_revision,
                        "tip": old_tip, "root": event["result"]["root"],
                    }
                    historical_store = _CatalogHeadStore(self.store, catalog, historical_head)
                    historical = CowWorkspace(historical_store, self.tokenizer).graph(
                        namespace_id, preview_chars=preview_chars)
                    historical_branch = historical["branches"][0]
                    for node in historical["nodes"]:
                        self._insert_graph_node(nodes, node)
                    previous_tips.append({
                        "branch_id": event["branch"],
                        "node_id": historical_branch["tip"],
                        "revision": old_revision,
                        "snapshot": old_tip,
                    })

            refcounts = defaultdict(set)
            for branch_id, path_ids in branch_paths.items():
                for node_id in path_ids:
                    refcounts[node_id].add(branch_id)
            for node_id, node in nodes.items():
                node["current_branch_count"] = len(refcounts[node_id])
            branches.sort(key=lambda item: (item["creation_order"], item["id"]))
            shared = self._common_prefix_length([logical_paths[item["id"]] for item in branches])
            shared_partials = set()
            for node_id, node in nodes.items():
                if node["role"] != "assistant":
                    continue
                referencing = refcounts[node_id]
                if len(referencing) < 2:
                    continue
                final_versions = set()
                for branch_id in referencing:
                    matching = [logical_id for logical_id in logical_paths[branch_id]
                                if nodes[logical_id]["message_id"] == node["message_id"]]
                    if matching:
                        final_versions.add(matching[-1])
                # An older common publication is an answer prefix only when
                # branches actually end that logical answer at different versions.
                if len(final_versions) > 1:
                    shared_partials.add(node["message_id"])
                    node["shared_partial"] = True
            return {
                "chat_id": namespace_id, "root_id": anchor,
                "branches": branches, "nodes": list(nodes.values()),
                "shared_prefix_messages": shared,
                "shared_partial_messages": len(shared_partials),
                "previous_tips": previous_tips, "previous_tips_supported": True,
                "previous_tips_requested": include_previous,
                "execution_mode": "cow-store-message-graph",
            }

    @staticmethod
    def _branch_display(catalog):
        first_ordinals = {}
        for event in catalog["events"]:
            name, ordinal = event["branch"], event["ordinal"]
            first_ordinals[name] = min(first_ordinals.get(name, ordinal), ordinal)
        ordered = sorted((head["name"] for head in catalog["heads"]),
                         key=lambda name: (first_ordinals.get(name, -1), name))
        human_names = {name for name in ordered
                       if not (re.fullmatch(r"b-op[0-9a-f]{32}", name)
                               or re.fullmatch(r"b-[0-9a-f]{32}", name))}
        used = set(human_names)
        result = {}
        for index, name in enumerate(ordered):
            ordinal = first_ordinals.get(name, -1)
            if name in human_names:
                display = name
            else:
                display = "Main" if index == 0 else f"Branch{index + 1}"
                if display in used:
                    display = f"Branch{index + 1}-{name[-4:]}"
                used.add(display)
            result[name] = {"creation_order": ordinal, "display_name": display}
        return result

    @staticmethod
    def _graph_node_id(material):
        raw = json.dumps(material, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return "message:" + hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _graph_node(node_id, parent_id, depth, message_id, role, preview,
                    message_increment, action_snapshot, content):
        return {
            "id": node_id, "parent_id": parent_id, "depth": depth,
            "role": role, "message_id": message_id, "preview": preview,
            "message_increment": message_increment,
            "current_branch_count": 0, "action_snapshot": action_snapshot,
            "boundary_available": action_snapshot is not None, "shared_partial": False,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _insert_graph_node(nodes, node):
        previous = nodes.get(node["id"])
        if previous is not None:
            for key in ("parent_id", "depth", "role", "message_id", "message_increment",
                        "action_snapshot", "boundary_available"):
                if previous[key] != node[key]:
                    raise IntegrityError("Canonical graph node identity has conflicting provenance")
            if (previous["preview"] != node["preview"] or
                    previous["content_sha256"] != node["content_sha256"]):
                raise IntegrityError("Canonical graph node identity has conflicting content")
            return
        nodes[node["id"]] = node

    def _assistant_prefix_text(self, message_id, descriptor, segments, endpoint, *, final):
        tokens = []
        for segment in segments:
            tokens.extend(segment["tokens"])
        parts = []
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        first_segment=segments[0]
        if first_segment["kind"]=="resume":
            try:
                decoder.setstate((bytes.fromhex(first_segment["source_controller"].get("output_pending_hex", "")),0))
            except (TypeError,ValueError) as error:
                raise IntegrityError("Assistant source decoder state is malformed") from error
        final_controller = segments[-1]["result_controller"]
        eos = final_controller.get("policy", {}).get("eos_tokens", [])
        stop_state=final_controller.get("text_stop")
        stops=stop_state.get("stop_sequences",()) if isinstance(stop_state,dict) else ()
        from context_time_travel.text_stops import TextStopMatcher
        source_stop=first_segment["source_controller"].get("text_stop")
        if (first_segment["kind"]=="resume" and isinstance(source_stop,dict)
                and tuple(source_stop.get("stop_sequences",()))==tuple(stops)):
            matcher=TextStopMatcher.restore(source_stop)
        else:
            matcher=TextStopMatcher(stops) if stops else None
        ended = False
        for token in tokens:
            if token in eos:
                if matcher is not None and not matcher.finished and not matcher.stopped:
                    result=matcher.feed(b"",final=True)
                    if result.safe_bytes:
                        value=decoder.decode(result.safe_bytes,final=False)
                        if value:parts.append(value)
                if final:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        parts.append(tail)
                ended = True
                break
            try:
                piece=self.tokenizer.token_bytes(token)
                if matcher is None:
                    value=decoder.decode(piece,final=False)
                    matched=None
                else:
                    matched=matcher.feed(piece)
                    value=decoder.decode(matched.safe_bytes,final=False)
            except Exception as error:
                raise IntegrityError("Tokenizer cannot decode a committed assistant token") from error
            if value:
                parts.append(value)
            if matched is not None and matched.stopped:
                if final:
                    tail=decoder.decode(b"",final=True)
                    if tail:parts.append(tail)
                ended=True
                break
        if final and not ended:
            if matcher is not None and not matcher.finished and not matcher.stopped:
                result=matcher.feed(b"",final=True)
                if result.safe_bytes:
                    value=decoder.decode(result.safe_bytes,final=False)
                    if value:parts.append(value)
            value = decoder.decode(b"", final=True)
            if value:
                parts.append(value)
        text = "".join(parts)
        return descriptor["text"] if final else text

    @staticmethod
    def _common_prefix_length(paths):
        if not paths:
            return 0
        count = 0
        shortest = min(map(len, paths))
        for index in range(shortest):
            value = paths[0][index]
            if any(path[index] != value for path in paths[1:]):
                break
            count += 1
        return count

    def _canonical_path(self, head):
        root_id, tokens = history_lineage(self.store,head['tip'])
        reverse_path = []
        seen = set()
        cursor = head["tip"]
        while cursor is not None:
            if cursor in seen:
                raise IntegrityError("Cycle in selected COW history")
            seen.add(cursor)
            record = self.store.get_snapshot(cursor)
            reverse_path.append((cursor, record))
            cursor = record["parent"]
        path = list(reversed(reverse_path))
        if not path or path[0][0] != root_id or path[-1][0] != head["tip"]:
            raise IntegrityError("Selected COW history has an invalid root or tip")
        concatenated = []
        positions = {}
        consumed=0
        for identity, record in path:
            if identity in positions:
                raise IntegrityError("Duplicate snapshot identity in selected history")
            concatenated.extend(record["inputs"])
            if isinstance(tokens,InputHistory):
                if record['schema']!=2:raise IntegrityError('Mixed history crossed input schemas')
                consumed+=len(InputHistory(record['inputs']))
            else:
                if record['schema']!=1:raise IntegrityError('Text history crossed input schemas')
                consumed+=len(record['inputs'])
            positions.setdefault(consumed, []).append(identity)
        combined=InputHistory(concatenated) if isinstance(tokens,InputHistory) else tuple(concatenated)
        if combined != tokens:
            raise IntegrityError("Snapshot path differs from canonical token lineage")
        return root_id, tokens, path, positions

    @staticmethod
    def _events_on_ancestry(catalog, ancestry, head):
        """Resolve annotations by snapshot and explicit branch provenance.

        Canonical snapshots can be content-addressed identically while two
        independent chat branches publish different display annotations. Walk
        captured fork source triples to constrain known branch histories. For
        legacy source-less forks, accept only a unique inherited annotation;
        fail closed rather than combining conflicting provenance.
        """
        branch_scope = {head["name"]: head["revision"]}
        branch_order = [head["name"]]
        current_branch = head["name"]
        visited = set()
        while current_branch not in visited:
            visited.add(current_branch)
            fork_events = [event for event in catalog["events"]
                           if event["branch"] == current_branch and event["type"] == "fork"]
            if not fork_events:
                break
            source = fork_events[0]["event"].get("source")
            if source is None:
                break
            source_branch = source.get("branch")
            source_revision = source.get("revision")
            source_tip = source.get("tip")
            if (not isinstance(source_branch, str) or type(source_revision) is not int
                    or not isinstance(source_tip, str) or source_revision < 0):
                raise IntegrityError("Fork presentation provenance is malformed")
            source_events = [event for event in catalog["events"]
                             if event["branch"] == source_branch
                             and event["result"]["revision"] == source_revision
                             and event["result"]["tip"] == source_tip]
            if len(source_events) != 1:
                raise IntegrityError("Fork presentation provenance does not identify one source revision")
            if source_branch in visited:
                raise IntegrityError("Cycle in fork presentation provenance")
            branch_scope[source_branch] = source_revision
            branch_order.append(source_branch)
            current_branch = source_branch

        by_snapshot = defaultdict(lambda: {"scoped": defaultdict(list), "fallback": []})
        for event in catalog["events"]:
            result = event["result"]
            presentation = event["presentation"]
            if (result["tip"] in ancestry and presentation is not None
                    and presentation["kind"] != "rewind"):
                if event["branch"] in branch_scope:
                    if result["revision"] <= branch_scope[event["branch"]]:
                        by_snapshot[result["tip"]]["scoped"][event["branch"]].append(event)
                    # A later publication on a known branch is outside the
                    # captured fork source and must never fall back into scope.
                    continue
                else:
                    by_snapshot[result["tip"]]["fallback"].append(event)
        selected = []
        for snapshot, groups in by_snapshot.items():
            scoped = groups["scoped"]
            # Prefer the selected branch, then its explicitly captured fork
            # ancestry. A branch may revisit the same content-addressed
            # snapshot at a later revision after a rewind; the latest receipt
            # within that branch's captured revision is the selected
            # publication. Historical heads therefore resolve against their
            # own revision ceiling instead of conflicting with future receipts.
            candidates = next((scoped[branch] for branch in branch_order
                               if scoped.get(branch)), None)
            if candidates is not None:
                latest_revision = max(event["result"]["revision"] for event in candidates)
                latest = [event for event in candidates
                          if event["result"]["revision"] == latest_revision]
                if len(latest) != 1:
                    raise IntegrityError(
                        f"Multiple scoped presentation receipts for snapshot {snapshot} "
                        f"at branch revision {latest_revision}"
                    )
                selected.append(latest[0])
                continue
            candidates = groups["fallback"]
            by_digest = defaultdict(list)
            for event in candidates:
                by_digest[event["digest"]].append(event)
            if len(by_digest) == 1:
                # Identical immutable descriptors do not conflict. Keep one
                # event in stable catalog order to avoid duplicate projection.
                selected.append(candidates[0])
                continue
            raise IntegrityError(
                f"Ambiguous presentation provenance for canonical snapshot {snapshot}"
            )
        yield from sorted(selected, key=lambda item: (item["ordinal"], item["branch"], item["operation_id"]))

    def _message_ids(self, catalog, head):
        root_id, _tokens, path, _positions = self._canonical_path(head)
        if root_id != head["root"]:
            raise IntegrityError("Catalog branch root differs from canonical lineage")
        ancestry = {identity for identity, _record in path}
        identities = set()
        for event in self._events_on_ancestry(catalog, ancestry, head):
            presentation = event["presentation"]
            if presentation is None:
                continue
            identities.update(message["id"] for message in presentation["messages"])
            output = presentation.get("output_span")
            if isinstance(output, dict):
                identities.add(output["message_id"])
        return identities

    def _project_messages(self, catalog, head, ancestry, token_ids, path, positions):
        records = {identity: record for identity, record in path}
        input_messages = {}
        assistant_groups = {}
        for event in self._events_on_ancestry(catalog, ancestry, head):
            presentation = event["presentation"]
            if presentation is None:
                continue
            kind = presentation["kind"]
            source_order = (event["ordinal"], event["branch"], event["operation_id"])
            if kind == "root":
                # The root annotation covers the whole rendered prompt. It has
                # no per-message token offsets, so no message gets a fake node.
                for index, source_message in enumerate(presentation["messages"]):
                    self._add_input_message(input_messages, source_message,
                        position=0, order=(*source_order, index), node_id=None,
                        boundary_reason="prepared_prompt_has_no_per_message_snapshot",
                        token_start=None, token_count=None)
            elif kind in ("initial_turn", "user_turn"):
                span = presentation["input_span"]
                end = span["start"] + span["count"]
                boundary_ids = positions.get(end, [])
                exact_boundary = (len(boundary_ids) == 1
                                  and end == len(token_ids)
                                  and event["result"]["tip"] == boundary_ids[0])
                node_id = boundary_ids[0] if exact_boundary else None
                for index, source_message in enumerate(presentation["messages"]):
                    self._add_input_message(input_messages, source_message,
                        position=span["start"], order=(*source_order, index), node_id=node_id,
                        boundary_reason=None if exact_boundary else "prompt_and_output_share_snapshot",
                        token_start=span["start"], token_count=span["count"])
            output = presentation.get("output_span")
            if isinstance(output, dict):
                result_tip = event["result"]["tip"]
                result_record = records.get(result_tip)
                if result_record is None:
                    raise IntegrityError("Presentation result is not on the selected branch ancestry")
                stream_key = result_record["controller"].get("stream_key")
                message_id = output["message_id"]
                group = assistant_groups.setdefault(message_id, {
                    "id": message_id,
                    "stream_key": stream_key,
                    "segments": [],
                    "first_order": (*source_order, output["start"]),
                })
                if group["stream_key"] != stream_key:
                    # A typed reseed is a zero-output controller transition
                    # that deliberately carries the same assistant identity
                    # into a new RNG stream. Accept only that exact bridge;
                    # ordinary output cannot silently cross stream identities.
                    if kind != "reseed" or output["count"] != 0:
                        raise IntegrityError("Assistant display ID is reused by a different generation stream")
                    group["stream_key"] = stream_key
                if source_order < group["first_order"][:3]:
                    group["first_order"] = (*source_order, output["start"])
                start = output["start"]
                end = start + output["count"]
                if start < 0 or end > len(token_ids):
                    raise IntegrityError("Assistant output span is outside canonical token history")
                source_snapshot = presentation.get("source", {}).get("snapshot")
                source_record = records.get(source_snapshot)
                if source_record is None:
                    raise IntegrityError("Assistant source is not on selected canonical ancestry")
                group["segments"].append({
                    "start": start,
                    "end": end,
                    "tokens": generated_text(history_segment(token_ids,start,end)),
                    "kind": kind,
                    "result_tip": result_tip,
                    "source_controller": source_record["controller"],
                    "result_controller": result_record["controller"],
                    "order": source_order,
                })

        inputs = list(input_messages.values())
        projected = inputs[:]
        for message_id, group in assistant_groups.items():
            assistant = self._decode_assistant(message_id, group)
            projected.append(assistant)
        projected.sort(key=lambda message: message["_order"])
        for message in projected:
            message.pop("_order", None)
        return projected

    @staticmethod
    def _add_input_message(messages, source, *, position, order, node_id,
                           boundary_reason, token_start, token_count):
        message = {
            "id": source["id"],
            "role": source["role"],
            "text": source["content"],
            "content": source["content"],
            "node_id": node_id,
            "boundary_available": node_id is not None,
            "boundary_reason": boundary_reason,
            "token_start": token_start,
            "token_count": token_count,
            "incomplete": False,
            "pending_utf8": False,
            "_order": (position, 0, *order),
        }
        previous = messages.get(source["id"])
        if previous is not None:
            if previous["role"] != message["role"] or previous["text"] != message["text"]:
                raise IntegrityError("One presentation message ID has conflicting content")
            # The same immutable root/message may be referenced by several
            # branches in one namespace. Retain its earliest canonical view.
            if message["_order"] < previous["_order"]:
                messages[source["id"]] = message
            return
        messages[source["id"]] = message

    def _decode_assistant(self, message_id, group):
        segments = sorted(group["segments"], key=lambda item: (item["start"], item["end"], item["order"]))
        unique = []
        for segment in segments:
            same_range = next((item for item in unique
                               if item["start"] == segment["start"] and item["end"] == segment["end"]), None)
            if same_range is not None:
                if same_range["tokens"] != segment["tokens"]:
                    raise IntegrityError("Forked assistant span differs at the same canonical token range")
                continue
            if unique and segment["start"] < unique[-1]["end"]:
                raise IntegrityError("Assistant output spans overlap inconsistently")
            if unique and segment["start"] > unique[-1]["end"]:
                raise IntegrityError("Assistant output spans contain an unexplained token gap")
            unique.append(segment)
        if not unique:
            raise IntegrityError("Assistant message has no published output span")
        first = unique[0]
        if first["kind"] == "resume":
            pending_hex = first["source_controller"].get("output_pending_hex", "")
        else:
            # A new answer/user turn initializes a new per-message decoder.
            pending_hex = ""
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            decoder.setstate((bytes.fromhex(pending_hex), 0))
        except (TypeError, ValueError) as error:
            raise IntegrityError("Assistant decoder state is malformed") from error
        parts = []
        final_tip = unique[-1]["result_tip"]
        final_controller = unique[-1]["result_controller"]
        eos = final_controller.get("policy", {}).get("eos_tokens", [])
        if not isinstance(eos, list) or any(type(token) is not int for token in eos):
            raise IntegrityError("Assistant controller EOS policy is malformed")
        stop_descriptor=final_controller.get("text_stop")
        stops=stop_descriptor.get("stop_sequences",()) if isinstance(stop_descriptor,dict) else ()
        from context_time_travel.text_stops import TextStopMatcher
        source_stop=first["source_controller"].get("text_stop")
        if (first["kind"]=="resume" and isinstance(source_stop,dict)
                and tuple(source_stop.get("stop_sequences",()))==tuple(stops)):
            matcher=TextStopMatcher.restore(source_stop)
        else:
            matcher=TextStopMatcher(stops) if stops else None
        ended = False
        for segment in unique:
            if segment is not first and segment["kind"] in ("resume", "reseed"):
                expected_pending = segment["source_controller"].get("output_pending_hex", "")
                if decoder.getstate()[0].hex() != expected_pending:
                    raise IntegrityError("Assistant UTF-8 decoder boundary differs from its source controller")
                expected_stop=segment["source_controller"].get("text_stop")
                if matcher is not None and expected_stop is not None and matcher.export_state()!=expected_stop:
                    raise IntegrityError("Assistant text-stop boundary differs from its source controller")
            for token in segment["tokens"]:
                if token in eos:
                    if matcher is not None and not matcher.finished and not matcher.stopped:
                        final=matcher.feed(b"",final=True)
                        if final.safe_bytes:
                            text=decoder.decode(final.safe_bytes,final=False)
                            if text:parts.append(text)
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        parts.append(tail)
                    ended = True
                    break
                try:
                    piece=self.tokenizer.token_bytes(token)
                    if matcher is None:
                        text=decoder.decode(piece,final=False)
                        matched=None
                    else:
                        matched=matcher.feed(piece)
                        text=decoder.decode(matched.safe_bytes,final=False)
                except Exception as error:
                    raise IntegrityError("Tokenizer cannot decode a committed assistant token") from error
                if text:
                    parts.append(text)
                if matched is not None and matched.stopped:
                    tail=decoder.decode(b"",final=True)
                    if tail:parts.append(tail)
                    ended=True
                    break
            if ended:
                break
        pending = decoder.getstate()[0]
        if not ended:
            expected_pending = final_controller.get("output_pending_hex", "")
            if pending.hex() != expected_pending:
                raise IntegrityError("Projected assistant UTF-8 state differs from durable controller")
            if matcher is not None and matcher.export_state()!=stop_descriptor:
                raise IntegrityError("Projected text-stop state differs from durable controller")
        elif matcher is not None and matcher.export_state()!=stop_descriptor:
            raise IntegrityError("Projected completed text-stop state differs from durable controller")
        final_status = final_controller.get("status")
        first_order = min((segment["start"], *segment["order"]) for segment in unique)
        return {
            "id": message_id,
            "role": "assistant",
            "text": "".join(parts),
            "content": "".join(parts),
            "node_id": final_tip,
            "boundary_available": True,
            "boundary_reason": None,
            "token_start": unique[0]["start"],
            "token_count": unique[-1]["end"] - unique[0]["start"],
            "snapshot_boundaries": list(dict.fromkeys(segment["result_tip"] for segment in unique)),
            "incomplete": final_status == "active",
            "pending_utf8": bool(pending) and not ended,
            "_order": (unique[0]["start"], 1, *first_order[1:]),
        }

    def _saved_points(self, catalog, head, ancestry):
        result = []
        for event in catalog["events"]:
            if event["branch"] != head["name"]:
                continue
            receipt = event["receipt"]
            publication = receipt["result"]
            snapshot_id = publication["tip"]
            result.append({
                "snapshot_id": snapshot_id,
                "node_id": snapshot_id,
                "branch_id": head["name"],
                "revision": publication["revision"],
                "operation_id": event["operation_id"],
                "kind": event["presentation"]["kind"] if event["presentation"] else event["type"],
                "source_snapshot": event["event"].get("from"),
                "on_selected_ancestry": snapshot_id in ancestry,
                "current": snapshot_id == head["tip"] and publication["revision"] == head["revision"],
                "boundary_available": True,
                "presentation_digest": event["digest"],
            })
        return result
