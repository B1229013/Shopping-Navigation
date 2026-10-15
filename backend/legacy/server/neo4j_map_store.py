"""neo4j_map_store.py — 把 TopoGraphV2 存入 / 讀出 Neo4j Aura，供多人共用。

使用方式：
    from server.neo4j_map_store import Neo4jMapStore

    store = Neo4jMapStore()          # 自動讀 .env
    store.upload("0908家樂福1")       # 把本地 topomap.json 上傳到 Neo4j
    topo = store.download("0908家樂福1")  # 從 Neo4j 下載回 TopoGraphV2
    store.close()
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

from server.topomap_v2 import TopoGraphV2


load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


class Neo4jMapStore:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self._uri = uri or os.getenv("NEO4J_URI", "")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 上傳：TopoGraphV2 → Neo4j ──────────────────────────────────

    def upload(self, place_name: str, topo: Optional[TopoGraphV2] = None,
               json_path: Optional[str] = None) -> int:
        """把一份 TopoGraphV2 寫入 Neo4j。

        可傳入 topo 物件，或傳 json_path 讓它自己讀檔。
        同一個 place_name 重複上傳會先清掉舊資料再寫入。
        回傳寫入的節點數。
        """
        if topo is None and json_path:
            topo = TopoGraphV2.load(json_path)
        if topo is None:
            raise ValueError("需要提供 topo 或 json_path")

        data = topo.to_dict()

        with self._driver.session(database=self._database) as session:
            # 清掉同場所的舊資料
            session.run(
                "MATCH (n) WHERE n.place_name = $place DETACH DELETE n",
                place=place_name,
            )

            # 寫入節點
            for node in data["nodes"]:
                props = _flatten_node(node, place_name, data)
                if node.get("ntype") == "photo":
                    session.run(
                        """
                        CREATE (p:Photo $props)
                        """,
                        props=props,
                    )
                else:
                    session.run(
                        """
                        CREATE (o:Object $props)
                        """,
                        props=props,
                    )

            # 寫入邊
            for edge in data["edges"]:
                etype = edge.get("etype", "")
                from_id = edge["from"]
                to_id = edge["to"]
                edge_props = _flatten_edge(edge)

                if etype == "walkway":
                    session.run(
                        """
                        MATCH (a {place_name: $place, node_id: $from_id})
                        MATCH (b {place_name: $place, node_id: $to_id})
                        CREATE (a)-[:WALKWAY $props]->(b)
                        """,
                        place=place_name, from_id=from_id, to_id=to_id,
                        props=edge_props,
                    )
                elif etype.startswith("contains_"):
                    session.run(
                        """
                        MATCH (a {place_name: $place, node_id: $from_id})
                        MATCH (b {place_name: $place, node_id: $to_id})
                        CREATE (a)-[:CONTAINS $props]->(b)
                        """,
                        place=place_name, from_id=from_id, to_id=to_id,
                        props=edge_props,
                    )
                elif etype == "adjacent":
                    session.run(
                        """
                        MATCH (a {place_name: $place, node_id: $from_id})
                        MATCH (b {place_name: $place, node_id: $to_id})
                        CREATE (a)-[:ADJACENT $props]->(b)
                        """,
                        place=place_name, from_id=from_id, to_id=to_id,
                        props=edge_props,
                    )
                elif etype == "same_object":
                    session.run(
                        """
                        MATCH (a {place_name: $place, node_id: $from_id})
                        MATCH (b {place_name: $place, node_id: $to_id})
                        CREATE (a)-[:SAME_OBJECT $props]->(b)
                        """,
                        place=place_name, from_id=from_id, to_id=to_id,
                        props=edge_props,
                    )

        return len(data["nodes"])

    # ── 下載：Neo4j → TopoGraphV2 ──────────────────────────────────

    def download(self, place_name: str) -> Optional[TopoGraphV2]:
        """從 Neo4j 讀回某場所的地圖，還原成 TopoGraphV2。"""
        with self._driver.session(database=self._database) as session:
            # 讀節點
            result = session.run(
                "MATCH (n) WHERE n.place_name = $place RETURN n",
                place=place_name,
            )
            nodes_raw = [dict(record["n"]) for record in result]
            if not nodes_raw:
                return None

            # 讀邊
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE a.place_name = $place AND b.place_name = $place
                RETURN a.node_id AS from_id, b.node_id AS to_id,
                       type(r) AS rel_type, properties(r) AS props
                """,
                place=place_name,
            )
            edges_raw = [dict(record) for record in result]

            # 讀地圖 metadata
            meta_result = session.run(
                """
                MATCH (n:Photo {place_name: $place})
                RETURN n.created_at AS created_at, n.updated_at AS updated_at,
                       n.current_position AS current_position,
                       n.position_history AS position_history
                ORDER BY n.node_id LIMIT 1
                """,
                place=place_name,
            )
            meta = meta_result.single()

        # 組裝回 to_dict() 格式
        nodes = []
        for n in nodes_raw:
            node = {"id": n.pop("node_id")}
            n.pop("place_name", None)
            # 還原 sensor_data（上傳時攤平成 sensor_ 前綴）
            sensor_data = {}
            remove_keys = []
            for k, v in n.items():
                if k.startswith("sensor_"):
                    sensor_data[k[len("sensor_"):]] = v
                    remove_keys.append(k)
            for k in remove_keys:
                n.pop(k)
            if sensor_data:
                n["sensor_data"] = sensor_data
            node.update(n)
            nodes.append(node)

        edges = []
        rel_type_to_etype = {
            "WALKWAY": "walkway",
            "ADJACENT": "adjacent",
            "SAME_OBJECT": "same_object",
        }
        for e in edges_raw:
            props = dict(e["props"])
            rel = e["rel_type"]
            if rel == "CONTAINS":
                etype = props.pop("etype", "contains_center")
            else:
                etype = rel_type_to_etype.get(rel, rel.lower())
            edge = {
                "from": e["from_id"],
                "to": e["to_id"],
                "key": props.pop("key", f"{etype}_{e['from_id']}_{e['to_id']}"),
                "etype": etype,
            }
            edge.update(props)
            edges.append(edge)

        # 從第一個 Photo 節點取 metadata
        current_position = None
        position_history = []
        created_at = ""
        updated_at = ""
        if meta:
            current_position = meta.get("current_position")
            ph = meta.get("position_history")
            if ph:
                position_history = json.loads(ph) if isinstance(ph, str) else list(ph)
            created_at = meta.get("created_at", "")
            updated_at = meta.get("updated_at", "")

        max_id = max((n["id"] for n in nodes), default=0)
        data = {
            "place_name": place_name,
            "created_at": created_at,
            "updated_at": updated_at,
            "next_id": max_id + 1,
            "current_position": current_position,
            "position_history": position_history,
            "nodes": nodes,
            "edges": edges,
        }
        return TopoGraphV2.from_dict(data)

    # ── 列出所有場所 ────────────────────────────────────────────────

    def list_places(self) -> List[Dict]:
        """列出 Neo4j 裡所有場所名稱及節點數。"""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.place_name IS NOT NULL
                RETURN n.place_name AS place, count(n) AS node_count
                ORDER BY place
                """
            )
            return [{"place_name": r["place"], "node_count": r["node_count"]}
                    for r in result]

    # ── 刪除某場所 ──────────────────────────────────────────────────

    def delete_place(self, place_name: str) -> int:
        """刪除某場所的所有節點和邊，回傳刪除的節點數。"""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (n) WHERE n.place_name = $place
                WITH n, count(n) AS cnt
                DETACH DELETE n
                RETURN cnt
                """,
                place=place_name,
            )
            record = result.single()
            return record["cnt"] if record else 0

    # ── 測試連線 ────────────────────────────────────────────────────

    def ping(self) -> str:
        with self._driver.session(database=self._database) as session:
            result = session.run("RETURN '連線成功！' AS msg")
            return result.single()["msg"]


# ── 工具函式 ────────────────────────────────────────────────────────

def _flatten_node(node: dict, place_name: str, map_data: dict) -> dict:
    """把巢狀的 node dict 攤平成 Neo4j 能接受的純屬性（不能有巢狀 dict/list）。"""
    props: dict = {
        "place_name": place_name,
        "node_id": node["id"],
        "ntype": node.get("ntype", ""),
    }

    # 照片節點的屬性
    for k in ("photo_path", "photo_file", "timestamp", "region",
              "is_virtual", "note", "label", "label_norm", "score",
              "position", "ocr_text", "role", "photo_id", "grid_cell"):
        if k in node and node[k] is not None:
            props[k] = node[k]

    # 把 sensor_data 攤平，加 sensor_ 前綴
    sensor = node.get("sensor_data")
    if isinstance(sensor, dict):
        for sk, sv in sensor.items():
            if sv is not None:
                props[f"sensor_{sk}"] = sv

    # box 轉成字串（Neo4j 不支援巢狀 list 屬性）
    box = node.get("box")
    if box:
        props["box"] = json.dumps(box)

    # 地圖級 metadata 存在第一個 Photo 節點上
    if node["id"] == 0 and node.get("ntype") == "photo":
        props["created_at"] = map_data.get("created_at", "")
        props["updated_at"] = map_data.get("updated_at", "")
        cp = map_data.get("current_position")
        if cp is not None:
            props["current_position"] = cp
        ph = map_data.get("position_history")
        if ph:
            props["position_history"] = json.dumps(ph)

    return props


def _flatten_edge(edge: dict) -> dict:
    """把邊的屬性攤平。"""
    props: dict = {}
    for k, v in edge.items():
        if k in ("from", "to"):
            continue
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            props[k] = json.dumps(v)
        else:
            props[k] = v
    return props
