create extension if not exists "uuid-ossp";

-- Rooms
create table if not exists rooms (
  id uuid primary key default uuid_generate_v4(),
  owner_tg_id bigint not null,
  title text not null,
  budget_min int,
  budget_max int,
  join_until timestamptz,
  draw_at timestamptz,
  deliver_until timestamptz,
  reveal_at timestamptz,
  geo text,
  rules_json jsonb default '{}'::jsonb,
  requires_address boolean default false,
  status text not null default 'active',
  created_at timestamptz not null default now()
);

-- Participants
create table if not exists participants (
  id uuid primary key default uuid_generate_v4(),
  room_id uuid not null references rooms(id) on delete cascade,
  user_tg_id bigint not null,
  name text not null,
  wishlist text[] default '{}',
  anti text[] default '{}',
  address_json jsonb default '{}'::jsonb,
  joined_at timestamptz not null default now(),
  unique (room_id, user_tg_id)
);

-- Exclusions
create table if not exists exclusions (
  id uuid primary key default uuid_generate_v4(),
  room_id uuid not null references rooms(id) on delete cascade,
  giver_tg_id bigint not null,
  receiver_tg_id bigint not null,
  unique (room_id, giver_tg_id, receiver_tg_id)
);

-- Draw versions
create table if not exists draws (
  id uuid primary key default uuid_generate_v4(),
  room_id uuid not null references rooms(id) on delete cascade,
  created_at timestamptz not null default now()
);

-- Pairs (linked to draw)
create table if not exists pairs (
  id uuid primary key default uuid_generate_v4(),
  room_id uuid not null references rooms(id) on delete cascade,
  draw_id uuid not null references draws(id) on delete cascade,
  giver_tg_id bigint not null,
  receiver_tg_id bigint not null,
  created_at timestamptz not null default now(),
  unique (room_id, draw_id, giver_tg_id)
);

-- Anonymous messages
create table if not exists messages_anon (
  id uuid primary key default uuid_generate_v4(),
  room_id uuid not null references rooms(id) on delete cascade,
  from_tg_id bigint not null,
  to_tg_id bigint not null,
  text text not null,
  created_at timestamptz not null default now()
);

-- Deliveries (placeholder)
create table if not exists deliveries (
  id uuid primary key default uuid_generate_v4(),
  room_id uuid not null references rooms(id) on delete cascade,
  giver_tg_id bigint not null,
  proof_url text,
  tracking text,
  status text not null default 'planned',
  updated_at timestamptz not null default now()
);

-- Payments (placeholder)
create table if not exists payments (
  id uuid primary key default uuid_generate_v4(),
  user_tg_id bigint not null,
  room_id uuid,
  plan text not null default 'FREE',
  provider text,
  invoice_id text,
  amount_cents int,
  status text not null default 'pending',
  created_at timestamptz not null default now()
);
