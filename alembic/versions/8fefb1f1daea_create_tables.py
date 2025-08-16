#
# Alembic migration script
#
"""
Revision ID: 8fefb1f1daea
Revises: 80d7b35f6349
Create Date: 2025-08-15 21:56:12.025190

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8fefb1f1daea'
down_revision = '80d7b35f6349'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### Create tables ###
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('username', sa.String(), nullable=False, unique=True, index=True),
        sa.Column('first_name', sa.String()),
        sa.Column('last_name', sa.String()),
        sa.Column('picture_url', sa.String()),
        sa.Column('user_type', sa.String(), nullable=True),
        sa.Column('has_evm_wallet', sa.Boolean(), server_default=sa.text('0'), nullable=False),
        sa.Column('has_solana_wallet', sa.Boolean(), server_default=sa.text('0'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    op.create_index('ix_users_username', 'users', ['username'], unique=True)

    op.create_table(
        'categories',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text()),
        sa.Column('image_url', sa.String()),
        sa.Column('count', sa.Integer(), server_default=sa.text('0'), nullable=False),
    )
    op.create_index('ix_categories_name', 'categories', ['name'], unique=True)

    op.create_table(
        'topics',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False, index=True),
        sa.Column('description', sa.Text()),
        sa.Column('image_url', sa.String()),
        sa.Column('slug', sa.String(), unique=True, index=True),
        sa.Column('is_public', sa.Boolean(), server_default=sa.text('1'), nullable=False),
        sa.Column('project_code', sa.String(), index=True),
        sa.Column('posts_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('language', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('owner_username', sa.String(), sa.ForeignKey('users.username'), nullable=True),
        sa.Column('category_id', sa.Integer(), sa.ForeignKey('categories.id'), nullable=True),
    )
    op.create_index('ix_topics_name', 'topics', ['name'], unique=False)
    op.create_index('ix_topics_slug', 'topics', ['slug'], unique=True)
    op.create_index('ix_topics_project_code', 'topics', ['project_code'], unique=False)

    op.create_table(
        'posts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('title', sa.String(), nullable=False, index=True),
        sa.Column('video_link', sa.String()),
        sa.Column('thumbnail_url', sa.String()),
        sa.Column('gif_thumbnail_url', sa.String(), nullable=True),
        sa.Column('is_available_in_public_feed', sa.Boolean(), server_default=sa.text('1'), nullable=False),
        sa.Column('is_locked', sa.Boolean(), server_default=sa.text('0'), nullable=False),
        sa.Column('slug', sa.String(), unique=True, index=True),
        sa.Column('identifier', sa.String(), unique=True, index=True),
        sa.Column('comment_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('upvote_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('view_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('exit_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('rating_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('average_rating', sa.Float(), server_default=sa.text('0'), nullable=False),
        sa.Column('share_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('bookmark_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('contract_address', sa.String(), nullable=True),
        sa.Column('chain_id', sa.String(), nullable=True),
        sa.Column('chart_url', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('category_id', sa.Integer(), sa.ForeignKey('categories.id'), nullable=False),
        sa.Column('topic_id', sa.Integer(), sa.ForeignKey('topics.id'), nullable=False),
        sa.Column('owner_username', sa.String(), sa.ForeignKey('users.username'), nullable=False),
    )
    op.create_index('ix_posts_title', 'posts', ['title'], unique=False)
    op.create_index('ix_posts_slug', 'posts', ['slug'], unique=True)
    op.create_index('ix_posts_identifier', 'posts', ['identifier'], unique=True)

    op.create_table(
        'tags',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False, unique=True, index=True),
    )
    op.create_index('ix_tags_name', 'tags', ['name'], unique=True)

    op.create_table(
        'post_tags',
        sa.Column('post_id', sa.Integer(), sa.ForeignKey('posts.id'), primary_key=True),
        sa.Column('tag_id', sa.Integer(), sa.ForeignKey('tags.id'), primary_key=True),
    )

    op.create_table(
        'user_interactions',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('interaction_type', sa.Enum('view', 'like', 'inspire', 'rating', name='interaction_type'), nullable=False),
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('username', sa.String(), sa.ForeignKey('users.username'), nullable=False),
        sa.Column('post_id', sa.Integer(), sa.ForeignKey('posts.id'), nullable=False),
    )


def downgrade() -> None:
    # ### Drop tables in reverse order due to FKs ###
    op.drop_table('user_interactions')
    op.drop_table('post_tags')
    op.drop_index('ix_tags_name', table_name='tags')
    op.drop_table('tags')
    op.drop_index('ix_posts_identifier', table_name='posts')
    op.drop_index('ix_posts_slug', table_name='posts')
    op.drop_index('ix_posts_title', table_name='posts')
    op.drop_table('posts')
    op.drop_index('ix_topics_project_code', table_name='topics')
    op.drop_index('ix_topics_slug', table_name='topics')
    op.drop_index('ix_topics_name', table_name='topics')
    op.drop_table('topics')
    op.drop_index('ix_categories_name', table_name='categories')
    op.drop_table('categories')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')
