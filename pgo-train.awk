# Generate a small synthetic workload for the instrumented build.
function draw(bound) {
    state = (state * 48271) % 2147483647
    return state % bound
}
function quoted(value) { return draw(5) == 0 ? "\"" value "\"" : value }
function event_csv(id, user, ts, ingest, kind, product, price) {
    return quoted(id) "," quoted(user) "," quoted(ts) "," quoted(ingest) "," quoted(kind) "," quoted(product) "," quoted(price)
}
function user_id(n) {
    if (n % 251 == 0) return "customer-" n
    return sprintf(n % 5 == 0 ? "u%07d" : "u%08d", n)
}
function clock_text(t, zone) {
    return sprintf("2025-01-%02dT%02d:%02d:%02d%s",
                   1 + int(t / 86400), int(t / 3600) % 24, int(t / 60) % 60, t % 60, zone)
}
function timestamp(t, form) {
    if (form < 6) return clock_text(t, "Z")
    if (form == 6) return clock_text(t + 7200, "+02:00")
    return sprintf("%.0f", (epoch + t) * 1000 + draw(1000))
}
BEGIN {
    state = 92417
    epoch = 1735689600
    OFS = ","
    print "{\"window_start\":\"2025-01-02T00:00:00Z\",\"window_end\":\"2025-01-16T00:00:00Z\"}" > "manifest.json"
    print "product_id,category" > "products.csv"
    for (i = 1; i <= 1600; ++i) print "p" i, "category" (i % 19) > "products.csv"
    print "campaign_id,channel" > "campaigns.csv"
    for (i = 1; i <= 200; ++i) print "c_" i, "channel" (i % 6) > "campaigns.csv"
    links = "links/part.csv"
    print "user_id_a,user_id_b" > links
    for (i = 2; i <= 22000; ++i) {
        if (i % 4) print user_id(i), user_id(i - 1) > links
    }
    for (i = 0; i < 4; ++i) {
        events[i] = sprintf("events/part-%d.csv", i)
        clicks[i] = sprintf("clicks/part-%d.csv", i)
        print "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents" > events[i]
        print "click_id,user_id,campaign_id,ts" > clicks[i]
    }
    for (i = 1; i <= 400000; ++i) {
        shard = draw(4)
        person = (draw(12) ? 1 + draw(26000) : 1 + draw(12))
        user = user_id(person)
        id = sprintf(i % 7 ? "e%08d" : "e%07d", i)
        if (i % 613 == 0) id = "long-event-" i
        t = 86400 + ((person * 863 + draw(16) * 38879) % (13 * 86400)) + draw(600)
        time_field = timestamp(t, draw(10))
        ingest = t + draw(90)
        type = (i % 37 == 0 ? "purchase" : (i % 9 == 0 ? "cart" : "view"))
        price = (type == "purchase" ? 100 + draw(45000) : "")
        product = "p" (1 + draw(1600))
        print event_csv(id, user, time_field, clock_text(ingest, "Z"), type, product, price) > events[shard]
        if (i % 31 == 0)
            print event_csv(id, user_id(person + 1), time_field, clock_text(ingest + 60, "Z"), type, product, price) > events[(shard + 1) % 4]
        if (i % 307 == 0)
            print event_csv(id, user, "invalid", clock_text(ingest - 1, "Z"), type, product, price) > events[(shard + 2) % 4]
        if (i % 223 == 0) print "bad,\"field,arity" > events[shard]
        if (i % 4 == 0)
            print quoted("click" i), quoted(user), quoted("c_" (1 + draw(200))), quoted(timestamp(t - draw(3600), draw(10))) > clicks[shard]
    }
}
