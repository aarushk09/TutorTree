create extension if not exists pgcrypto;

create table if not exists public.human_evaluations (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null,
    scenario_id text not null,
    selected_intervention text not null check (selected_intervention in ('A', 'B', 'Tie')),
    reasoning text,
    created_at timestamptz not null default now()
);

alter table public.human_evaluations enable row level security;

revoke all on public.human_evaluations from anon;
revoke all on public.human_evaluations from authenticated;
grant insert on public.human_evaluations to anon;
grant all on public.human_evaluations to service_role;

drop policy if exists "human_evaluations_anon_insert" on public.human_evaluations;

create policy "human_evaluations_anon_insert"
on public.human_evaluations
for insert
to anon
with check (
    session_id is not null
    and scenario_id is not null
    and length(btrim(scenario_id)) > 0
    and selected_intervention in ('A', 'B', 'Tie')
    and coalesce(length(reasoning), 0) <= 4000
);

comment on table public.human_evaluations is 'Blind A/B human preference evaluations for anonymous AI tutoring intervention sessions.';

create table if not exists public.human_feedback (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null,
    feedback_text text,
    created_at timestamptz not null default now()
);

alter table public.human_feedback enable row level security;

revoke all on public.human_feedback from anon;
revoke all on public.human_feedback from authenticated;
grant insert on public.human_feedback to anon;
grant all on public.human_feedback to service_role;

drop policy if exists "human_feedback_anon_insert" on public.human_feedback;

create policy "human_feedback_anon_insert"
on public.human_feedback
for insert
to anon
with check (
    session_id is not null
    and coalesce(length(feedback_text), 0) <= 8000
);

comment on table public.human_feedback is 'Anonymous final open-ended feedback for the blind AI tutoring intervention study.';
