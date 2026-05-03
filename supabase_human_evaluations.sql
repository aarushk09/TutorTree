create extension if not exists pgcrypto;

create table if not exists public.human_evaluations (
    id uuid primary key default gen_random_uuid(),
    session_id uuid not null,
    scenario_id text not null,
    selected_intervention text not null check (selected_intervention in ('A', 'B', 'Tie')),
    evaluator_role text,
    reasoning text,
    response_time_ms integer,
    warning_label text,
    created_at timestamptz not null default now()
);

alter table public.human_evaluations
add column if not exists evaluator_role text;

alter table public.human_evaluations
add column if not exists response_time_ms integer,
add column if not exists warning_label text;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'human_evaluations_evaluator_role_check'
          and conrelid = 'public.human_evaluations'::regclass
    ) then
        alter table public.human_evaluations
        add constraint human_evaluations_evaluator_role_check
        check (
            evaluator_role is null
            or evaluator_role in (
                'University Faculty',
                'K-12 Teacher',
                'Student',
                'EdTech Researcher',
                'Other'
            )
        );
    end if;

    if not exists (
        select 1
        from pg_constraint
        where conname = 'human_evaluations_response_time_ms_check'
          and conrelid = 'public.human_evaluations'::regclass
    ) then
        alter table public.human_evaluations
        add constraint human_evaluations_response_time_ms_check
        check (response_time_ms is null or response_time_ms >= 0);
    end if;

    if not exists (
        select 1
        from pg_constraint
        where conname = 'human_evaluations_warning_label_check'
          and conrelid = 'public.human_evaluations'::regclass
    ) then
        alter table public.human_evaluations
        add constraint human_evaluations_warning_label_check
        check (warning_label is null or warning_label in ('answered_under_3_seconds'));
    end if;
end $$;

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
    and evaluator_role in (
        'University Faculty',
        'K-12 Teacher',
        'Student',
        'EdTech Researcher',
        'Other'
    )
    and coalesce(length(reasoning), 0) <= 4000
    and (response_time_ms is null or response_time_ms >= 0)
    and (warning_label is null or warning_label = 'answered_under_3_seconds')
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
