#!/usr/bin/env bash
set -euo pipefail
export GOFLAGS=-mod=vendor
export GOTOOLCHAIN=local
export GOPROXY=off
export GOSUMDB=off
export GOWORK=off
export GOCACHE=/tmp/urlooker-recovery-cache
export HOME=/tmp/urlooker-recovery-home
export TZ=UTC
mkdir -p "$HOME" "$GOCACHE" /tmp/urlooker-recovery-build output/recovery/streams

# The adapter uses the pinned application's key and consistent-hash wrapper.
# Its input paths are interpreted from the checkout root.
cat > /tmp/urlooker-recovery-build/route.go <<'GO'
package main

import (
    "encoding/json"
    "fmt"
    "io"
    "os"
    "sort"

    "github.com/710leo/urlooker/dataobj"
    "github.com/710leo/urlooker/modules/web/g"
    "github.com/710leo/urlooker/modules/web/sender"
    "gopkg.in/yaml.v2"
)

type Request struct {
    Config string `json:"config"`
    Item dataobj.ItemStatus `json:"item"`
}
type Result struct {
    PK string `json:"pk"`
    Address string `json:"address"`
}
type Routing struct {
    ring *sender.ConsistentHashNodeRing
    cluster map[string]string
}

func run() error {
    routes := make(map[string]Routing)
    dec, enc := json.NewDecoder(os.Stdin), json.NewEncoder(os.Stdout)
    for {
        var req Request
        err := dec.Decode(&req)
        if err == io.EOF { return nil }
        if err != nil { return err }
        routing, exists := routes[req.Config]
        if !exists {
            buf, err := os.ReadFile(req.Config)
            if err != nil { return err }
            var cfg g.GlobalConfig
            if err := yaml.Unmarshal(buf, &cfg); err != nil { return err }
            if cfg.Alarm == nil || !cfg.Alarm.Enable || len(cfg.Alarm.Cluster) == 0 {
                return fmt.Errorf("no active alarm cluster in %s", req.Config)
            }
            names := make([]string, 0, len(cfg.Alarm.Cluster))
            for name := range cfg.Alarm.Cluster { names = append(names, name) }
            sort.Strings(names)
            routing = Routing{
                ring: sender.NewConsistentHashNodeRing(cfg.Alarm.Replicas, names),
                cluster: cfg.Alarm.Cluster,
            }
            routes[req.Config] = routing
        }
        pk := req.Item.PK()
        node, err := routing.ring.GetNode(pk)
        if err != nil { return err }
        address, ok := routing.cluster[node]
        if !ok { return fmt.Errorf("unknown ring node %q", node) }
        if err := enc.Encode(Result{PK: pk, Address: address}); err != nil { return err }
    }
}
func main() {
    if err := run(); err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
}
GO

go build -o /tmp/urlooker-recovery-build/route /tmp/urlooker-recovery-build/route.go
go build -o /tmp/urlooker-recovery-build/replay ./recovery/tools/replay

python3 - <<'PY'
import collections
import csv
import json
import pathlib
import subprocess

root = pathlib.Path('recovery')
out = pathlib.Path('output/recovery')

def load_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)

def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def encode_line(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n'

def write_lines(path, values):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for value in values:
            f.write(encode_line(value))

batches = csv_rows(root / 'outbox/batches.csv')
batch_sequences = [int(row['seq']) for row in batches]
if len(set(batch_sequences)) != len(batch_sequences):
    raise RuntimeError('duplicate dispatch sequence')

receipts = {}
with (root / 'outbox/receipts.jsonl').open(encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        seq = row['seq']
        if seq in receipts:
            raise RuntimeError('conflicting receipt')
        if row['status'] not in ('acked', 'not_sent', 'reply_lost'):
            raise RuntimeError('unknown receipt status')
        receipts[seq] = row['status']
if set(batch_sequences) != set(receipts):
    raise RuntimeError('batch and receipt inventory differ')

schedule = sorted(load_json(root / 'topology/schedule.json'), key=lambda x: x['first_seq'])
receivers = sorted(load_json(root / 'topology/receivers.json'), key=lambda x: x['name'])
by_address = {row['address']: row['name'] for row in receivers}
streams = {row['name']: [] for row in receivers}

records = []
for row in csv_rows(root / 'outbox/items.csv'):
    seq, index = int(row['seq']), int(row['position'])
    if seq not in receipts:
        raise RuntimeError('item belongs to an unlisted batch')
    item = {field: int(row[field]) for field in ('id', 'sid', 'resp_time', 'push_time', 'result')}
    item['ip'] = row['ip']
    item['resp_code'] = row['resp_code']
    effective = [epoch for epoch in schedule if epoch['first_seq'] <= seq]
    if not effective:
        raise RuntimeError('no topology for item')
    records.append({'seq': seq, 'index': index, 'item': item, 'config': effective[-1]['config']})
records.sort(key=lambda row: (row['seq'], row['index']))
positions = collections.defaultdict(list)
for row in records:
    positions[row['seq']].append(row['index'])
for seq, indices in positions.items():
    if indices != list(range(len(indices))):
        raise RuntimeError(f'invalid item positions for sequence {seq}')

route_input = ''.join(encode_line({'config': row['config'], 'item': row['item']}) for row in records)
routed = subprocess.run(
    ['/tmp/urlooker-recovery-build/route'],
    input=route_input, text=True, encoding='utf-8', stdout=subprocess.PIPE, check=True,
)
route_results = [json.loads(line) for line in routed.stdout.splitlines()]
if len(route_results) != len(records):
    raise RuntimeError('incomplete route output')

dispatch = []
for row, route in zip(records, route_results):
    status = receipts[row['seq']]
    name = None if status == 'not_sent' else by_address[route['address']]
    dispatch.append({
        'seq': row['seq'], 'index': row['index'], 'receipt': status,
        'pk': route['pk'], 'receiver': name,
    })
    if name is not None:
        streams[name].append({'seq': row['seq'], 'index': row['index'], 'item': row['item']})
write_lines(out / 'dispatch.jsonl', dispatch)

emissions = []
receiver_summaries = []
for receiver in receivers:
    name = receiver['name']
    stream_path = out / 'streams' / (name + '.jsonl')
    write_lines(stream_path, streams[name])
    with stream_path.open(encoding='utf-8') as stream:
        replay = subprocess.run(
            ['/tmp/urlooker-recovery-build/replay', 'recovery/strategies.json'],
            stdin=stream, stdout=subprocess.PIPE, text=True, encoding='utf-8', check=True,
        )
    node_emissions = []
    for line in replay.stdout.splitlines():
        event = json.loads(line)
        event['receiver'] = name
        node_emissions.append(event)
    emissions.extend(node_emissions)
    counts = collections.Counter(row['event']['status'] for row in node_emissions)
    receiver_summaries.append({
        'name': name,
        'address': receiver['address'],
        'items': len(streams[name]),
        'problem_events': counts['PROBLEM'],
        'ok_events': counts['OK'],
    })

emissions.sort(key=lambda row: (row['seq'], row['index']))
write_lines(out / 'events.jsonl', emissions)
receipt_counts = collections.Counter(receipts.values())
summary = {
    'receipts': {status: receipt_counts[status] for status in ('acked', 'not_sent', 'reply_lost')},
    'uncertain_sequences': sorted(seq for seq, status in receipts.items() if status == 'reply_lost'),
    'receivers': receiver_summaries,
}
with (out / 'summary.json').open('w', encoding='utf-8', newline='\n') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write('\n')
PY
